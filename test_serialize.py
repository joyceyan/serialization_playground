#!/usr/bin/env python3
"""
Correctness tests for serialization schemes (SOTA format).

Usage:
    source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate
    python test_serialize.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

from serialize import (
    HAS_ZSTD,
    decode_baseline,
    decode_transpose_v1,
    encode_baseline,
    encode_transpose_v1,
    load_and_convert,
    mlx_to_sota_format,
)

ARTIFACT_DIR = "/Users/jyan/src/parameter-golf-fork/logs"
DEFAULT_ARTIFACT = os.path.join(
    ARTIFACT_DIR,
    "f3ffa88d-62a7-4575-8c89-0ef6a07eb11a_mlx_model.int8.ptz",
)

PASS = 0
FAIL = 0


def check(name: str, condition: bool, msg: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {msg}")


def roundtrip_check(
    scheme_name: str,
    encode_fn,
    decode_fn,
    sota_obj: dict,
    max_allowed_error: float = 0.0,
) -> None:
    """Test that encode → decode produces identical (or acceptably close) output."""
    print(f"\n--- {scheme_name} ---")

    blob = encode_fn(sota_obj)
    check("encode produces bytes", isinstance(blob, bytes) and len(blob) > 0)

    decoded = decode_fn(blob)
    check("decode produces dict", isinstance(decoded, dict))
    check("has 'w' key", "w" in decoded)
    check("has 'm' key", "m" in decoded)

    orig_w = sota_obj["w"]
    decoded_w = decoded.get("w", {})
    orig_m = sota_obj["m"]
    decoded_m = decoded.get("m", {})

    # Check all tensors in w
    for key in orig_w:
        check(f"w/{key} present", key in decoded_w, f"missing from decoded")
        if key not in decoded_w:
            continue
        orig_np = orig_w[key].numpy() if isinstance(orig_w[key], torch.Tensor) else np.asarray(orig_w[key])
        rt_np = decoded_w[key].numpy() if isinstance(decoded_w[key], torch.Tensor) else np.asarray(decoded_w[key])
        check(f"w/{key} shape", orig_np.shape == rt_np.shape,
              f"shape mismatch: {orig_np.shape} vs {rt_np.shape}")
        if orig_np.shape == rt_np.shape:
            max_err = float(np.abs(orig_np.astype(np.float32) - rt_np.astype(np.float32)).max())
            check(f"w/{key} roundtrip (max_err={max_err:.6f})",
                  max_err <= max_allowed_error,
                  f"max_err={max_err:.6f} > {max_allowed_error}")

    # Check metadata
    check("metadata keys match", set(orig_m.keys()) == set(decoded_m.keys()),
          f"orig={set(orig_m.keys())} decoded={set(decoded_m.keys())}")
    for key in orig_m:
        if key in decoded_m:
            check(f"m/{key} matches", orig_m[key] == decoded_m[key],
                  f"{orig_m[key]} vs {decoded_m[key]}")


def test_synthetic() -> None:
    """Test with a synthetic small artifact in SOTA format."""
    print("\n=== Synthetic SOTA format tests ===")

    sota_obj = {
        "w": {
            "blocks.0.attn.c_q.weight.q": torch.randint(-20, 20, size=(64, 32), dtype=torch.int8),
            "blocks.0.attn.c_q.weight.scale": torch.from_numpy(np.random.uniform(0.001, 0.1, size=(64,)).astype(np.float16)),
            "blocks.0.resid_mix": torch.randn(2, 32, dtype=torch.float32),
        },
        "m": {
            "blocks.0.attn.c_q.weight": {"type": "int6"},
            "blocks.0.resid_mix": "passthrough_ctrl",
        },
    }

    if HAS_ZSTD:
        roundtrip_check("baseline (synthetic)", encode_baseline, decode_baseline, sota_obj)
        roundtrip_check("transpose_v1 (synthetic)", encode_transpose_v1, decode_transpose_v1, sota_obj)
    else:
        print("  SKIP: zstandard not installed")


def test_sota_conversion() -> None:
    """Test MLX → SOTA format conversion."""
    print("\n=== MLX → SOTA conversion tests ===")

    if not os.path.exists(DEFAULT_ARTIFACT):
        print(f"  SKIP: artifact not found at {DEFAULT_ARTIFACT}")
        return

    sota_obj = load_and_convert(DEFAULT_ARTIFACT)
    check("conversion produces dict", isinstance(sota_obj, dict))
    check("has 'w' key", "w" in sota_obj)
    check("has 'm' key", "m" in sota_obj)

    w = sota_obj["w"]
    m = sota_obj["m"]

    # Check int6 tensors have correct range
    for name, info in m.items():
        if isinstance(info, dict) and info.get("type") == "int6":
            t = w[name + ".q"]
            lo, hi = int(t.min()), int(t.max())
            check(f"{name} int6 range", lo >= -32 and hi <= 31,
                  f"range [{lo}, {hi}] exceeds int6 [-32, 31]")

    # Check int8 tensors exist
    for name, info in m.items():
        if isinstance(info, dict) and info.get("type") == "int8":
            check(f"{name} has .q", name + ".q" in w)
            check(f"{name} has .scale", name + ".scale" in w)

    # tok_emb should be int8 (needs full range)
    check("tok_emb is int8", m.get("tok_emb.weight", {}) == {"type": "int8"},
          f"got {m.get('tok_emb.weight')}")


def test_real_roundtrip() -> None:
    """Test full roundtrip with a real artifact."""
    print("\n=== Real artifact roundtrip tests ===")

    if not os.path.exists(DEFAULT_ARTIFACT):
        print(f"  SKIP: artifact not found at {DEFAULT_ARTIFACT}")
        return

    if not HAS_ZSTD:
        print("  SKIP: zstandard not installed")
        return

    sota_obj = load_and_convert(DEFAULT_ARTIFACT)
    roundtrip_check("baseline (real)", encode_baseline, decode_baseline, sota_obj)
    roundtrip_check("transpose_v1 (real)", encode_transpose_v1, decode_transpose_v1, sota_obj)


def main() -> None:
    test_synthetic()
    test_sota_conversion()
    test_real_roundtrip()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

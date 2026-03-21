#!/usr/bin/env python3
"""
Correctness tests for serialization schemes.

Usage:
    source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate
    python test_serialize.py
"""

from __future__ import annotations

import os
import sys
import numpy as np

from serialize import (
    HAS_ZSTD,
    decode_baseline,
    decode_zstd22,
    encode_baseline,
    encode_zstd22,
    load_artifact,
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
    quant_obj: dict,
    max_allowed_error: float = 0.0,
) -> None:
    """Test that encode → decode produces identical (or acceptably close) output."""
    print(f"\n--- {scheme_name} ---")

    blob = encode_fn(quant_obj)
    check(f"encode produces bytes", isinstance(blob, bytes) and len(blob) > 0)

    decoded = decode_fn(blob)
    check(f"decode produces dict", isinstance(decoded, dict))

    # Check all quantized tensors
    for key in quant_obj.get("quantized", {}):
        orig = np.asarray(quant_obj["quantized"][key])
        check(f"quantized/{key} present", key in decoded.get("quantized", {}),
              f"missing from decoded")
        if key not in decoded.get("quantized", {}):
            continue
        rt = np.asarray(decoded["quantized"][key])
        check(f"quantized/{key} shape", orig.shape == rt.shape,
              f"shape mismatch: {orig.shape} vs {rt.shape}")
        if orig.shape == rt.shape:
            max_err = float(np.abs(orig.astype(np.float32) - rt.astype(np.float32)).max())
            check(f"quantized/{key} roundtrip (max_err={max_err:.6f})",
                  max_err <= max_allowed_error,
                  f"max_err={max_err:.6f} > {max_allowed_error}")

    # Check all scales
    for key in quant_obj.get("scales", {}):
        orig = np.asarray(quant_obj["scales"][key])
        check(f"scales/{key} present", key in decoded.get("scales", {}))
        if key not in decoded.get("scales", {}):
            continue
        rt = np.asarray(decoded["scales"][key])
        max_err = float(np.abs(orig.astype(np.float32) - rt.astype(np.float32)).max())
        check(f"scales/{key} roundtrip (max_err={max_err:.6f})",
              max_err <= max_allowed_error)

    # Check all passthrough tensors
    for key in quant_obj.get("passthrough", {}):
        orig = np.asarray(quant_obj["passthrough"][key])
        check(f"passthrough/{key} present", key in decoded.get("passthrough", {}))
        if key not in decoded.get("passthrough", {}):
            continue
        rt = np.asarray(decoded["passthrough"][key])
        max_err = float(np.abs(orig.astype(np.float32) - rt.astype(np.float32)).max())
        check(f"passthrough/{key} roundtrip (max_err={max_err:.6f})",
              max_err <= max_allowed_error)

    # Check metadata preserved
    for meta_key in ["dtypes", "qmeta", "passthrough_orig_dtypes"]:
        if meta_key in quant_obj:
            check(f"{meta_key} preserved", meta_key in decoded,
                  f"missing from decoded")
            if meta_key in decoded:
                check(f"{meta_key} matches", quant_obj[meta_key] == decoded[meta_key],
                      f"content mismatch")


def test_synthetic() -> None:
    """Test with a synthetic small artifact."""
    print("\n=== Synthetic artifact tests ===")

    quant_obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": {
            "layer.weight": np.random.randint(-20, 20, size=(64, 32), dtype=np.int8),
        },
        "scales": {
            "layer.weight": np.random.uniform(0.001, 0.1, size=(64,)).astype(np.float16),
        },
        "dtypes": {"layer.weight": "bfloat16"},
        "passthrough": {
            "bias": np.random.randn(32).astype(np.float32),
        },
        "qmeta": {"layer.weight": {"scheme": "per_row", "axis": 0}},
        "passthrough_orig_dtypes": {},
    }

    roundtrip_check("baseline (synthetic)", encode_baseline, decode_baseline, quant_obj)
    if HAS_ZSTD:
        roundtrip_check("zstd-22 (synthetic)", encode_zstd22, decode_zstd22, quant_obj)


def test_real_artifact() -> None:
    """Test with a real artifact from training."""
    print("\n=== Real artifact tests ===")

    if not os.path.exists(DEFAULT_ARTIFACT):
        print(f"  SKIP: artifact not found at {DEFAULT_ARTIFACT}")
        return

    quant_obj = load_artifact(DEFAULT_ARTIFACT)
    roundtrip_check("baseline (real)", encode_baseline, decode_baseline, quant_obj)
    if HAS_ZSTD:
        roundtrip_check("zstd-22 (real)", encode_zstd22, decode_zstd22, quant_obj)


def main() -> None:
    test_synthetic()
    test_real_artifact()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

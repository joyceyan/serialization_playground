#!/usr/bin/env python3
"""
Correctness tests for serialization schemes.

Usage:
    cd /Users/jyan/src/serialization_playground
    python test_serialize.py [--model PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from serialize import (
    HAS_ZSTD,
    decode_baseline,
    encode_baseline,
    load_and_quantize,
    mixed_quantize_int6,
)

DEFAULT_MODEL = "/Users/jyan/src/serialization_playground/final_model.pt"

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
    quant_result: dict,
    quant_meta: dict,
    max_allowed_error: float = 0.0,
) -> None:
    print(f"\n--- {scheme_name} ---")

    blob = encode_fn(quant_result, quant_meta)
    check("encode produces bytes", isinstance(blob, bytes) and len(blob) > 0)

    decoded_w, decoded_m = decode_fn(blob)
    check("decode produces dicts", isinstance(decoded_w, dict) and isinstance(decoded_m, dict))

    for key in quant_result:
        check(f"{key} present", key in decoded_w, f"missing from decoded")
        if key not in decoded_w:
            continue
        orig = quant_result[key]
        rt = decoded_w[key]
        orig_np = orig.numpy() if isinstance(orig, torch.Tensor) else np.asarray(orig)
        rt_np = rt.numpy() if isinstance(rt, torch.Tensor) else np.asarray(rt)
        check(f"{key} shape", orig_np.shape == rt_np.shape,
              f"{orig_np.shape} vs {rt_np.shape}")
        if orig_np.shape == rt_np.shape:
            max_err = float(np.abs(orig_np.astype(np.float32) - rt_np.astype(np.float32)).max())
            check(f"{key} roundtrip (err={max_err:.6f})",
                  max_err <= max_allowed_error,
                  f"err={max_err} > {max_allowed_error}")

    check("metadata keys match",
          set(quant_meta.keys()) == set(decoded_m.keys()),
          f"orig={len(quant_meta)} decoded={len(decoded_m)}")


def test_synthetic() -> None:
    print("\n=== Synthetic tests ===")

    sd = {
        "blocks.0.attn.c_q.weight": torch.randn(512, 512),
        "blocks.0.mlp.fc.weight": torch.randn(1536, 512),
        "tok_emb.weight": torch.randn(1024, 512),
        "blocks.0.attn.q_gain": torch.ones(8),
        "blocks.0.attn_scale": torch.ones(512),
    }
    quant_result, quant_meta = mixed_quantize_int6(sd, {"mlp", "attn"})

    if HAS_ZSTD:
        roundtrip_check("baseline (synthetic)", encode_baseline, decode_baseline,
                       quant_result, quant_meta)
    else:
        print("  SKIP: zstandard not installed")


def test_real(model_path: str) -> None:
    print("\n=== Real H100 artifact tests ===")

    if not os.path.exists(model_path):
        print(f"  SKIP: {model_path} not found (pull from RunPod first)")
        return

    if not HAS_ZSTD:
        print("  SKIP: zstandard not installed")
        return

    quant_result, quant_meta = load_and_quantize(model_path)
    roundtrip_check("baseline (real)", encode_baseline, decode_baseline,
                   quant_result, quant_meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    test_synthetic()
    test_real(args.model)

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

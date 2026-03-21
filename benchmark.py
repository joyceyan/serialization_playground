#!/usr/bin/env python3
"""
Benchmark serialization schemes against a real H100 model artifact.

Usage:
    cd /Users/jyan/src/serialization_playground
    python benchmark.py [--model PATH]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

from serialize import (
    HAS_ZSTD,
    decode_baseline,
    encode_baseline,
    load_and_quantize,
    measure_scheme,
)

DEFAULT_MODEL = "/Users/jyan/src/serialization_playground/final_model.pt"


def artifact_summary(quant_result: dict, quant_meta: dict) -> None:
    """Print a summary of the quantized artifact."""
    total_int6 = 0
    total_int8 = 0
    total_scale = 0
    total_pass = 0
    n_int6 = 0
    n_int8 = 0

    for name, info in quant_meta.items():
        if isinstance(info, dict) and info.get("type") == "int6":
            t = quant_result[name + ".q"]
            total_int6 += t.nelement() * t.element_size()
            total_scale += quant_result[name + ".scale"].nelement() * quant_result[name + ".scale"].element_size()
            n_int6 += 1
        elif isinstance(info, dict) and info.get("type") == "int8":
            t = quant_result[name + ".q"]
            total_int8 += t.nelement() * t.element_size()
            total_scale += quant_result[name + ".scale"].nelement() * quant_result[name + ".scale"].element_size()
            n_int8 += 1
        elif info in ("passthrough", "passthrough_ctrl"):
            t = quant_result[name]
            total_pass += t.nelement() * t.element_size()

    total = total_int6 + total_int8 + total_scale + total_pass
    print(f"Quantized: {n_int6} int6 tensors ({total_int6:,}b), "
          f"{n_int8} int8 tensors ({total_int8:,}b), "
          f"scales {total_scale:,}b, passthrough {total_pass:,}b")
    print(f"Total raw bytes: {total:,}")

    # Value ranges
    print("\nInt6 tensor ranges:")
    for name, info in sorted(quant_meta.items()):
        if isinstance(info, dict) and info.get("type") == "int6":
            t = quant_result[name + ".q"]
            lo, hi = int(t.min()), int(t.max())
            n_zero = int((t == 0).sum())
            pct_zero = 100.0 * n_zero / t.nelement()
            print(f"  {name}: [{lo}, {hi}] {pct_zero:.1f}% zeros")

    print("\nInt8 tensor ranges:")
    for name, info in sorted(quant_meta.items()):
        if isinstance(info, dict) and info.get("type") == "int8":
            t = quant_result[name + ".q"]
            lo, hi = int(t.min()), int(t.max())
            n_zero = int((t == 0).sum())
            pct_zero = 100.0 * n_zero / t.nelement()
            print(f"  {name}: [{lo}, {hi}] {pct_zero:.1f}% zeros")
    print()


def print_table(results: list[dict]) -> None:
    headers = ["scheme", "compressed", "savings", "max_err", "encode_ms", "decode_ms"]
    baseline_size = results[0]["compressed_bytes"] if results else 1

    rows = []
    for r in results:
        savings = baseline_size - r["compressed_bytes"]
        pct = 100.0 * savings / baseline_size if baseline_size > 0 else 0
        rows.append([
            r["name"],
            f'{r["compressed_bytes"]:,}',
            f'{savings:+,} ({pct:+.1f}%)',
            f'{r["max_abs_error"]:.6f}',
            f'{r["encode_ms"]:.0f}',
            f'{r["decode_ms"]:.0f}',
        ])

    widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to final_model.pt")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Pull final_model.pt from your RunPod pod first.")
        sys.exit(1)

    print(f"Loading & quantizing: {os.path.basename(args.model)}")
    quant_result, quant_meta = load_and_quantize(args.model)

    if args.summary:
        artifact_summary(quant_result, quant_meta)
        return

    artifact_summary(quant_result, quant_meta)

    if not HAS_ZSTD:
        print("ERROR: zstandard not installed")
        sys.exit(1)

    schemes = [
        ("baseline_zstd22", encode_baseline, decode_baseline),
    ]

    print("Running benchmarks (3 trials each)...\n")
    results = []
    for name, enc, dec in schemes:
        print(f"  {name}...", end=" ", flush=True)
        r = measure_scheme(name, enc, dec, quant_result, quant_meta)
        print(f"{r['compressed_bytes']:,} bytes")
        results.append(r)

    print()
    print_table(results)


if __name__ == "__main__":
    main()

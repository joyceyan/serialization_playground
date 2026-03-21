#!/usr/bin/env python3
"""
Benchmark all serialization schemes against real artifacts in SOTA format.

Usage:
    source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate
    python benchmark.py [--artifact PATH] [--summary]
"""

from __future__ import annotations

import argparse
import glob
import math
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
    measure_scheme,
)

ARTIFACT_DIR = "/Users/jyan/src/parameter-golf-fork/logs"
DEFAULT_ARTIFACT = os.path.join(
    ARTIFACT_DIR,
    "f3ffa88d-62a7-4575-8c89-0ef6a07eb11a_mlx_model.int8.ptz",
)


def print_table(results: list[dict]) -> None:
    """Print results as a formatted table."""
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


def artifact_summary(sota_obj: dict) -> None:
    """Print a summary of the SOTA-format artifact structure."""
    w = sota_obj.get("w", {})
    m = sota_obj.get("m", {})

    total_int6_q = 0
    total_int8_q = 0
    total_scale = 0
    total_pass = 0
    n_int6 = 0
    n_int8 = 0

    def _nbytes(t):
        if isinstance(t, torch.Tensor):
            return t.nelement() * t.element_size()
        return np.asarray(t).nbytes

    for name, info in m.items():
        if isinstance(info, dict) and info.get("type") == "int6":
            total_int6_q += _nbytes(w[name + ".q"])
            total_scale += _nbytes(w[name + ".scale"])
            n_int6 += 1
        elif isinstance(info, dict) and info.get("type") == "int8":
            total_int8_q += _nbytes(w[name + ".q"])
            total_scale += _nbytes(w[name + ".scale"])
            n_int8 += 1
        elif info in ("passthrough", "passthrough_ctrl"):
            total_pass += _nbytes(w[name])

    total = total_int6_q + total_int8_q + total_scale + total_pass
    print(f"SOTA format: {n_int6} int6 tensors ({total_int6_q:,}b), "
          f"{n_int8} int8 tensors ({total_int8_q:,}b), "
          f"scales {total_scale:,}b, passthrough {total_pass:,}b")
    print(f"Total raw bytes: {total:,}")

    # Value ranges for quantized tensors
    print("\nInt6 tensor ranges:")
    for name, info in sorted(m.items()):
        if isinstance(info, dict) and info.get("type") == "int6":
            t = w[name + ".q"]
            lo, hi = int(t.min()), int(t.max())
            span = hi - lo + 1
            bits = math.ceil(math.log2(max(span, 1)))
            print(f"  {name}: [{lo}, {hi}] span={span} bits={bits}")

    print("\nInt8 tensor ranges:")
    for name, info in sorted(m.items()):
        if isinstance(info, dict) and info.get("type") == "int8":
            t = w[name + ".q"]
            lo, hi = int(t.min()), int(t.max())
            span = hi - lo + 1
            bits = math.ceil(math.log2(max(span, 1)))
            print(f"  {name}: [{lo}, {hi}] span={span} bits={bits}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark serialization schemes (SOTA format)")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Path to MLX .int8.ptz artifact")
    parser.add_argument("--summary", action="store_true", help="Print artifact summary only")
    args = parser.parse_args()

    if not os.path.exists(args.artifact):
        artifacts = sorted(glob.glob(os.path.join(ARTIFACT_DIR, "*.int8.ptz")))
        if artifacts:
            args.artifact = artifacts[-1]
            print(f"Using: {args.artifact}")
        else:
            print("No artifacts found. Run a smoke test first.")
            sys.exit(1)

    print(f"Loading & converting: {os.path.basename(args.artifact)}")
    sota_obj = load_and_convert(args.artifact)

    if args.summary:
        artifact_summary(sota_obj)
        return

    artifact_summary(sota_obj)

    if not HAS_ZSTD:
        print("ERROR: zstandard not installed. Install with: pip install zstandard")
        sys.exit(1)

    # Register all schemes to benchmark
    schemes = [
        ("baseline_zstd22", encode_baseline, decode_baseline),
        ("transpose_v1", encode_transpose_v1, decode_transpose_v1),
    ]

    # Run benchmarks
    print("Running benchmarks (3 trials each)...\n")
    results = []
    for name, enc, dec in schemes:
        print(f"  {name}...", end=" ", flush=True)
        r = measure_scheme(name, enc, dec, sota_obj)
        print(f"{r['compressed_bytes']:,} bytes")
        results.append(r)

    print()
    print_table(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Benchmark all serialization schemes against real artifacts.

Usage:
    source /Users/jyan/src/parameter-golf-fork/.venv/bin/activate
    python benchmark.py [--artifact PATH]
"""

from __future__ import annotations

import argparse
import glob
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
    measure_scheme,
)

ARTIFACT_DIR = "/Users/jyan/src/parameter-golf-fork/logs"
# Default to the most recent artifact
DEFAULT_ARTIFACT = os.path.join(
    ARTIFACT_DIR,
    "f3ffa88d-62a7-4575-8c89-0ef6a07eb11a_mlx_model.int8.ptz",
)


def print_table(results: list[dict]) -> None:
    """Print results as a formatted table."""
    headers = ["scheme", "compressed", "ratio", "max_err", "encode_ms", "decode_ms"]
    # Get baseline size for ratio calculation
    baseline_size = results[0]["compressed_bytes"] if results else 1

    rows = []
    for r in results:
        ratio = baseline_size / r["compressed_bytes"]
        rows.append([
            r["name"],
            f'{r["compressed_bytes"]:,}',
            f'{ratio:.3f}x',
            f'{r["max_abs_error"]:.6f}',
            f'{r["encode_ms"]:.0f}',
            f'{r["decode_ms"]:.0f}',
        ])

    # Column widths
    widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def artifact_summary(quant_obj: dict) -> None:
    """Print a summary of the artifact structure."""
    total_q = sum(np.asarray(a).nbytes for a in quant_obj.get("quantized", {}).values())
    total_s = sum(np.asarray(a).nbytes for a in quant_obj.get("scales", {}).values())
    total_p = sum(np.asarray(a).nbytes for a in quant_obj.get("passthrough", {}).values())
    n_tensors = len(quant_obj.get("quantized", {}))

    print(f"Artifact: {n_tensors} quantized tensors, {total_q:,} bytes int8, "
          f"{total_s:,} bytes scales, {total_p:,} bytes passthrough")
    print(f"Total raw tensor bytes: {total_q + total_s + total_p:,}")

    # Value range analysis
    print("\nValue ranges:")
    for name in sorted(quant_obj.get("quantized", {}).keys()):
        arr = np.asarray(quant_obj["quantized"][name])
        lo, hi = int(arr.min()), int(arr.max())
        bits_needed = max(lo.bit_length() + 1, hi.bit_length() + (1 if hi >= 0 else 0))
        # Simpler: how many bits to represent the range
        span = hi - lo + 1
        import math
        bits = math.ceil(math.log2(max(span, 1)))
        print(f"  {name}: [{lo}, {hi}] span={span} needs {bits} bits")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark serialization schemes")
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="Path to .int8.ptz artifact")
    parser.add_argument("--summary", action="store_true", help="Print artifact summary only")
    args = parser.parse_args()

    if not os.path.exists(args.artifact):
        print(f"Artifact not found: {args.artifact}")
        # Try to find any artifact
        artifacts = sorted(glob.glob(os.path.join(ARTIFACT_DIR, "*.int8.ptz")))
        if artifacts:
            args.artifact = artifacts[-1]
            print(f"Using: {args.artifact}")
        else:
            print("No artifacts found. Run a smoke test first.")
            sys.exit(1)

    print(f"Loading artifact: {os.path.basename(args.artifact)}")
    quant_obj = load_artifact(args.artifact)

    if args.summary:
        artifact_summary(quant_obj)
        return

    artifact_summary(quant_obj)

    # Register all schemes to benchmark
    schemes = [
        ("baseline_zlib9", encode_baseline, decode_baseline),
    ]
    if HAS_ZSTD:
        schemes.append(("zstd_22", encode_zstd22, decode_zstd22))

    # Run benchmarks
    print("Running benchmarks (3 trials each)...\n")
    results = []
    for name, enc, dec in schemes:
        print(f"  {name}...", end=" ", flush=True)
        r = measure_scheme(name, enc, dec, quant_obj)
        print(f"{r['compressed_bytes']:,} bytes")
        results.append(r)

    print()
    print_table(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test and benchmark serialization schemes.

Runs correctness checks (roundtrip accuracy) and benchmarks (size, timing)
against either a real H100 artifact or synthetic data.

Usage:
    cd /Users/jyan/src/serialization_playground
    python test_serialize.py [--model PATH] [--summary]
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
    decode_experiment,
    encode_baseline,
    encode_experiment,
    load_and_quantize,
    measure_scheme,
    mixed_quantize_int6,
)

DEFAULT_MODEL = "/Users/jyan/src/serialization_playground/final_model.pt"


# ==============================================================================
# ARTIFACT SUMMARY
# ==============================================================================

def artifact_summary(quant_result: dict, quant_meta: dict) -> None:
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


# ==============================================================================
# BENCHMARK TABLE
# ==============================================================================

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


# ==============================================================================
# TESTS
# ==============================================================================

def test_synthetic() -> None:
    """Correctness test with synthetic data (always runs, no model file needed)."""
    print("=== Synthetic correctness test ===\n")

    sd = {
        "blocks.0.attn.c_q.weight": torch.randn(512, 512),
        "blocks.0.attn.c_k.weight": torch.randn(256, 512),
        "blocks.0.mlp.fc.weight": torch.randn(1536, 512),
        "blocks.0.mlp.proj.weight": torch.randn(512, 1536),
        "tok_emb.weight": torch.randn(1024, 512),
        "blocks.0.attn.q_gain": torch.ones(8),
        "blocks.0.attn_scale": torch.ones(512),
        "blocks.0.mlp_scale": torch.ones(512),
        "blocks.0.resid_mix": torch.stack([torch.ones(512), torch.zeros(512)]),
    }
    quant_result, quant_meta = mixed_quantize_int6(sd, {"mlp", "attn"})

    if not HAS_ZSTD:
        print("SKIP: zstandard not installed")
        return

    schemes = [
        ("baseline_zstd22", encode_baseline, decode_baseline),
        ("experiment", encode_experiment, decode_experiment),
    ]

    passed = 0
    failed = 0
    for name, enc, dec in schemes:
        r = measure_scheme(name, enc, dec, quant_result, quant_meta, n_trials=1)
        if r["max_abs_error"] == 0.0:
            print(f"  PASS: {name} — {r['compressed_bytes']:,} bytes, lossless roundtrip")
            passed += 1
        else:
            print(f"  FAIL: {name} — max_err={r['max_abs_error']}")
            failed += 1

    print(f"\nSynthetic: {passed} passed, {failed} failed\n")
    return failed == 0


def test_real(model_path: str) -> None:
    """Full test + benchmark with real H100 artifact."""
    print(f"=== Real artifact: {os.path.basename(model_path)} ===\n")

    print("Loading & quantizing...")
    quant_result, quant_meta = load_and_quantize(model_path)
    artifact_summary(quant_result, quant_meta)

    if not HAS_ZSTD:
        print("ERROR: zstandard not installed")
        sys.exit(1)

    schemes = [
        ("baseline_zstd22", encode_baseline, decode_baseline),
        ("experiment", encode_experiment, decode_experiment),
    ]

    print("Running benchmarks (3 trials each)...\n")
    results = []
    for name, enc, dec in schemes:
        print(f"  {name}...", end=" ", flush=True)
        r = measure_scheme(name, enc, dec, quant_result, quant_meta)
        status = "OK" if r["max_abs_error"] == 0.0 else f"FAIL (err={r['max_abs_error']})"
        print(f"{r['compressed_bytes']:,} bytes — {status}")
        results.append(r)

    print()
    print_table(results)

    failed = sum(1 for r in results if r["max_abs_error"] > 0)
    if failed:
        print(f"\n{failed} scheme(s) failed roundtrip check!")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test and benchmark serialization schemes")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to final_model.pt")
    parser.add_argument("--summary", action="store_true", help="Print artifact summary only")
    args = parser.parse_args()

    # Always run synthetic test
    test_synthetic()

    # Run real test if model exists
    if os.path.exists(args.model):
        if args.summary:
            quant_result, quant_meta = load_and_quantize(args.model)
            artifact_summary(quant_result, quant_meta)
        else:
            test_real(args.model)
    else:
        print(f"Skipping real artifact test: {args.model} not found")
        print("Pull final_model.pt from RunPod to enable.")


if __name__ == "__main__":
    main()

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
import json
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

try:
    import brotli as _brotli
    HAS_BROTLI = True
except ImportError:
    _brotli = None
    HAS_BROTLI = False


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
# EXPERIMENT: K-means clustered ANS + LZMA
# ==============================================================================

def _sorted_keys(d):
    return sorted(d.keys())


def _derive_shape(key, arch_params):
    """Derive tensor shape from key name and architecture params dict."""
    D = arch_params["D"]
    H = arch_params["H"]
    KV = arch_params["KV"]
    MLP = arch_params["MLP"]
    L = arch_params["L"]
    vocab = arch_params["vocab"]
    bigram_dim = arch_params["bigram_dim"]
    bigram_vocab = arch_params.get("bigram_vocab", vocab * 2)
    ve_dim = arch_params.get("ve_dim", 0)
    hd = D // H
    kv = KV * hd
    mlp = D * MLP
    if key.endswith(".q"):
        base = key[:-2]
        if "c_q" in base: return [D, D]
        if "c_k" in base: return [kv, D]
        if "c_v" in base: return [kv, D]
        if ".attn." in base and "proj" in base: return [D, D]
        if "mlp.fc" in base: return [mlp, D]
        if ".mlp." in base and "proj" in base: return [D, mlp]
        if "tok_emb" in base: return [vocab, D]
        if "bigram.embed" in base: return [bigram_vocab, bigram_dim]
        if "ve_shared.embed" in base and ve_dim: return [vocab, ve_dim]
        if "ve_shared.proj" in base and ve_dim: return [kv, ve_dim]
        if "lm_head" in base: return [vocab, D]
    if key.endswith(".scale"):
        base = key[:-6]
        if "c_q" in base: return [D]
        if "c_k" in base: return [kv]
        if "c_v" in base: return [kv]
        if ".attn." in base and "proj" in base: return [D]
        if "mlp.fc" in base: return [mlp]
        if ".mlp." in base and "proj" in base: return [D]
        if "tok_emb" in base: return [vocab]
        if "bigram.embed" in base: return [bigram_vocab]
        if "ve_shared.embed" in base and ve_dim: return [vocab]
        if "ve_shared.proj" in base and ve_dim: return [kv]
        if "lm_head" in base: return [vocab]
    if "q_gain" in key: return [H]
    if "attn_scale" in key: return [D]
    if "mlp_scale" in key: return [D]
    if "resid_mix" in key: return [2, D]
    if "skip_weight" in key: return [L // 2, D]
    if "bigram.proj" in key: return [D, bigram_dim]
    if "bigram.scale" in key: return []
    if "smear" in key and "gate" in key: return [D]
    if "dtg_gate.weight" in key: return [1, D]
    if "dtg_gate.bias" in key: return [1]
    if "ve_shared.proj" in key and ve_dim: return [kv, ve_dim]
    if "ve_layer_scales" in key: return [1]
    if "ve_shared.scale" in key: return []
    return []


def encode_experiment(quant_result: dict[str, Tensor], quant_meta: dict[str, object],
                      arch_params: dict | None = None) -> bytes:
    """Separate int8/fp16/fp32 into distinct streams, transpose 2D, zstd-22 each."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    comp = zstandard.ZstdCompressor(level=22)

    # Classify tensors into streams by dtype
    int8_keys = []
    fp16_keys = []
    fp32_keys = []
    for k in _sorted_keys(quant_result):
        t = quant_result[k]
        if t.dtype == torch.int8:
            int8_keys.append(k)
        elif t.dtype == torch.float16:
            fp16_keys.append(k)
        else:
            fp32_keys.append(k)

    # Build raw byte streams
    streams = {}
    import constriction
    from scipy.cluster.vq import kmeans2

    # Separate int6 and int8 tensors
    int6_keys = []
    int8_only_keys = []
    for k in int8_keys:
        base = k[:-2] if k.endswith(".q") else k
        info = quant_meta.get(base)
        if isinstance(info, dict) and info.get("type") == "int6":
            int6_keys.append(k)
        else:
            int8_only_keys.append(k)

    # === INT6: K-means clustered row models ===
    N_CLUSTERS = 16
    int6_row_data = []      # zigzag int32 arrays per row
    int6_row_dists = []     # per-row probability distributions
    int6_row_tensor_idx = [] # which tensor each row belongs to
    int6_tensor_nrows = []  # rows per tensor

    for ti, k in enumerate(int6_keys):
        t = quant_result[k].contiguous().numpy()
        if t.ndim != 2:
            # 1D tensor — treat as single row
            arr = t.astype(np.int16).flatten()
            zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_row_tensor_idx.append(ti)
            int6_tensor_nrows.append(1)
            continue
        nrows = t.shape[0]
        int6_tensor_nrows.append(nrows)
        for i in range(nrows):
            row = t[i].astype(np.int16)
            zigzag = ((row << 1) ^ (row >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_row_tensor_idx.append(ti)

    # K-means clustering on sqrt(probs) — Hellinger distance is better for distributions
    dist_matrix = np.sqrt(np.array(int6_row_dists))
    centroids, labels = kmeans2(dist_matrix, N_CLUSTERS, minit="random", iter=100, seed=286)

    # Build per-cluster frequency tables
    cluster_freq_u16 = []
    cluster_models = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if not mask.any():
            cluster_freq_u16.append(np.ones(64, dtype=np.uint16))
            cluster_models.append(None)
            continue
        cluster_data = np.concatenate([int6_row_data[i] for i in range(len(labels)) if labels[i] == c])
        freqs = np.bincount(cluster_data, minlength=64)[:64].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        cluster_freq_u16.append(freq_u16)
        cluster_models.append(constriction.stream.model.Categorical(probs, perfect=False))

    # Single-stream ANS with model switching per row (encode in reverse)
    encoder = constriction.stream.stack.AnsCoder()
    for i in range(len(int6_row_data) - 1, -1, -1):
        c = labels[i]
        encoder.encode_reverse(int6_row_data[i], cluster_models[c])
    int6_compressed = encoder.get_compressed().tobytes()

    # Compress labels (1 byte per label for K > 16, 4-bit packed for K <= 16)
    if N_CLUSTERS <= 16:
        packed_labels = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
        for i in range(0, len(labels) - 1, 2):
            packed_labels[i // 2] = (labels[i] << 4) | labels[i + 1]
        if len(labels) % 2:
            packed_labels[-1] = labels[-1] << 4
    else:
        packed_labels = labels.astype(np.uint8)
    freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    labels_compressed = lzma.compress(packed_labels.tobytes(), format=lzma.FORMAT_RAW, filters=freq_filters)

    # Compress frequency tables
    all_freq_bytes = b"".join(f.tobytes() for f in cluster_freq_u16)
    freq_compressed = lzma.compress(all_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters)

    # === INT8 (non-int6): per-tensor ANS ===
    int8_parts = []
    for k in int8_only_keys:
        t = quant_result[k].contiguous()
        arr = t.numpy().astype(np.int16).flatten()
        zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
        freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(zigzag, model)
        int8_parts.append((encoder.get_compressed().tobytes(), freq_u16, len(zigzag)))

    # Pack everything into int8 stream
    # Format: [version(1)] [n_clusters(1)] [n_int6_tensors(2)] [n_int8_tensors(2)]
    #         [n_rows(4)] [freq_tables_len(2)] [freq_tables_lzma]
    #         [labels_len(2)] [labels_lzma]
    #         per-cluster: [c_len(4)] [c_data]
    #         [int6_tensor_nrows as packed bytes]
    #         per-int8-tensor: [numel(4)] [freq_table(512)] [c_len(4)] [c_data]
    ans_out = struct.pack("<BBHHI",
        1,  # version
        N_CLUSTERS,
        len(int6_keys),
        len(int8_only_keys),
        len(labels),
    )
    ans_out += struct.pack("<H", len(freq_compressed)) + freq_compressed
    ans_out += struct.pack("<H", len(labels_compressed)) + labels_compressed
    # Single ANS stream for all int6 data
    ans_out += struct.pack("<I", len(int6_compressed)) + int6_compressed
    # Tensor row counts derivable from architecture params — not stored
    # Int8 tensors
    int8_freq_bytes = b"".join(f.tobytes() for _, f, _ in int8_parts)
    int8_freq_c = lzma.compress(int8_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters) if int8_parts else b""
    ans_out += struct.pack("<H", len(int8_freq_c)) + int8_freq_c
    for c_data_bytes, _, numel in int8_parts:
        ans_out += struct.pack("<II", numel, len(c_data_bytes)) + c_data_bytes
    streams["int8"] = ans_out

    # fp16 stream: byte-shuffle (separate high and low bytes)
    # Order: passthrough tensors first (better LZMA compression)
    fp16_ordered = sorted(fp16_keys, key=lambda k: (0 if ".scale" not in k else 1, k))
    fp16_parts = []
    for k in fp16_ordered:
        t = quant_result[k].contiguous()
        fp16_parts.append(t.numpy().tobytes())
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        high = arr[0::2].tobytes()  # high bytes
        low = arr[1::2].tobytes()   # low bytes
        # Use raw LZMA2 with lp=0 (slightly better than lp=1 on this mixed data)
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
        streams["fp16"] = lzma.compress(high + low, format=lzma.FORMAT_RAW, filters=filters)

    # fp32 stream: byte-shuffle (4 sub-streams)
    fp32_parts = []
    for k in fp32_keys:
        t = quant_result[k]
        if t.ndim == 2:
            t = t.t().contiguous()
        fp32_parts.append(t.numpy().tobytes())
    if fp32_parts:
        raw = b"".join(fp32_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = b"".join(arr[i::4].tobytes() for i in range(4))
        streams["fp32"] = comp.compress(shuffled)

    # Encode metadata — robust format from fork
    all_keys = _sorted_keys(quant_result)
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys:
                meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set:
            pass  # skip .scale if matching .q exists
        elif k not in meta_keys:
            meta_keys.append(k)
    type_str = ""
    for mk in meta_keys:
        info = quant_meta.get(mk, "passthrough")
        if info == "passthrough": type_str += "p"
        elif info == "passthrough_ctrl": type_str += "c"
        elif isinstance(info, dict) and info.get("type") == "int6": type_str += "6"
        elif isinstance(info, dict) and info.get("type") == "int8": type_str += "8"
        else: type_str += "p"

    # Build dtype string (explicit, more robust than suffix-based derivation)
    dtype_str = ""
    for k in all_keys:
        t = quant_result[k]
        if t.dtype == torch.int8: dtype_str += "i"
        elif t.dtype == torch.float16: dtype_str += "h"
        else: dtype_str += "f"

    # Architecture params — use dict for flexibility
    if arch_params is None:
        arch_params = {"D": 512, "H": 8, "KV": 4, "MLP": 3, "L": 11,
                       "vocab": 1024, "bigram_dim": 128}

    header = json.dumps({
        "keys": all_keys,
        "arch": arch_params,
        "types": type_str,
        "dtypes": dtype_str,
    }, separators=(",", ":")).encode()
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_c = lzma.compress(header, format=lzma.FORMAT_RAW, filters=header_filters)

    # Pack: [header_len(2)] [header_compressed] [int8_len(4)] [int8_compressed] [fp16_len(4)] [fp16_compressed] [fp32_len(4)?] [fp32_compressed?]
    out = struct.pack("<H", len(header_c)) + header_c
    for label in ["int8", "fp16"]:
        blob = streams.get(label, b"")
        out += struct.pack("<I", len(blob)) + blob
    # Only include fp32 block if non-empty
    fp32_blob = streams.get("fp32", b"")
    if fp32_blob:
        out += struct.pack("<I", len(fp32_blob)) + fp32_blob
    return out


def decode_experiment(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Decode separate-stream format."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    decomp = zstandard.ZstdDecompressor()

    off = 0
    def read_block():
        nonlocal off
        sz = struct.unpack_from("<I", blob, off)[0]
        off += 4
        data = blob[off:off + sz]
        off += sz
        return data

    # Header uses 2-byte length prefix
    header_sz = struct.unpack_from("<H", blob, off)[0]
    off += 2
    raw_header = blob[off:off + header_sz]
    off += header_sz
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_json = json.loads(lzma.decompress(raw_header, format=lzma.FORMAT_RAW, filters=header_filters))

    all_keys = header_json["keys"]
    # Architecture params — support both dict and list formats
    arch = header_json["arch"]
    if isinstance(arch, list):
        arch = {"D": arch[0], "H": arch[1], "KV": arch[2], "MLP": arch[3], "L": arch[4],
                "vocab": arch[5] if len(arch) > 5 else 1024,
                "bigram_dim": arch[6] if len(arch) > 6 else 128}
    shapes_list = [_derive_shape(k, arch) for k in all_keys]

    # Dtype classification — use explicit dtype string if available, else derive from suffix
    dtype_str = header_json.get("dtypes", "")
    if dtype_str:
        int8_indices = [i for i, c in enumerate(dtype_str) if c == "i"]
        fp16_indices = [i for i, c in enumerate(dtype_str) if c == "h"]
        fp32_indices = [i for i, c in enumerate(dtype_str) if c == "f"]
    else:
        int8_indices = [i for i, k in enumerate(all_keys) if k.endswith(".q")]
        fp16_indices = [i for i, k in enumerate(all_keys) if not k.endswith(".q")]
        fp32_indices = []

    # Reconstruct quant_meta from type_str
    type_str = header_json.get("types", "")
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys:
                meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set:
            pass
        elif k not in meta_keys:
            meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}

    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "meta": quant_meta,
    }
    # Decode clustered ANS int8 stream
    import constriction
    int8_block = read_block()
    if header["int8_keys"] and int8_block:
        boff = 0
        version, n_clusters, n_int6, n_int8, n_rows = struct.unpack_from("<BBHHI", int8_block, boff)
        boff += 10
        freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]

        # Frequency tables
        freq_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
        all_freq_bytes = lzma.decompress(int8_block[boff:boff + freq_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
        boff += freq_c_len

        # Labels
        labels_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
        labels_packed = lzma.decompress(int8_block[boff:boff + labels_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
        boff += labels_c_len
        # Unpack labels
        labels_arr = np.frombuffer(labels_packed, dtype=np.uint8)
        if n_clusters <= 16:
            labels = np.empty(n_rows, dtype=np.int32)
            for i in range(0, n_rows - 1, 2):
                labels[i] = labels_arr[i // 2] >> 4
                labels[i + 1] = labels_arr[i // 2] & 0x0F
            if n_rows % 2:
                labels[n_rows - 1] = labels_arr[n_rows // 2] >> 4
        else:
            labels = labels_arr[:n_rows].astype(np.int32)

        # Parse cluster frequency tables
        cluster_probs = []
        for c in range(n_clusters):
            freq_u16 = np.frombuffer(all_freq_bytes[c * 128:(c + 1) * 128], dtype=np.uint16).astype(np.float64)
            probs = freq_u16 / freq_u16.sum()
            cluster_probs.append(probs)

        # Single ANS stream for all int6 data
        int6_c_len = struct.unpack_from("<I", int8_block, boff)[0]; boff += 4
        int6_c_bytes = int8_block[boff:boff + int6_c_len]; boff += int6_c_len

        # Derive tensor row counts from shapes
        int6_q_keys = [k for k in header["int8_keys"]
                       if isinstance(header["meta"].get(k[:-2] if k.endswith(".q") else k), dict)
                       and header["meta"].get(k[:-2] if k.endswith(".q") else k, {}).get("type") == "int6"]
        tensor_nrows = []
        for k in int6_q_keys:
            shape = header["shapes"][k]
            tensor_nrows.append(shape[0] if len(shape) >= 2 else 1)

        row_ncols = []
        for ti, nr in enumerate(tensor_nrows):
            shape = header["shapes"][int6_q_keys[ti]]
            ncols = shape[-1] if len(shape) >= 2 else int(np.prod(shape))
            for _ in range(nr):
                row_ncols.append(ncols)

        # Decode single ANS stream row by row with per-row cluster model
        cluster_models_dec = [constriction.stream.model.Categorical(p, perfect=False) for p in cluster_probs]
        int6_decoder = constriction.stream.stack.AnsCoder(np.frombuffer(int6_c_bytes, dtype=np.uint32))
        int6_decoded = []
        for row_idx in range(n_rows):
            c = int(labels[row_idx])
            ncols = row_ncols[row_idx]
            row_zigzag = int6_decoder.decode(cluster_models_dec[c], ncols)
            decoded = ((row_zigzag.astype(np.int16) >> 1) ^ -(row_zigzag.astype(np.int16) & 1)).astype(np.int8)
            int6_decoded.append(decoded.tobytes())

        # Group decoded rows back into tensors
        int6_tensor_bytes = []
        row_cursor = 0
        for nr in tensor_nrows:
            tensor_rows = b"".join(int6_decoded[row_cursor:row_cursor + nr])
            int6_tensor_bytes.append(tensor_rows)
            row_cursor += nr

        # Decode int8 (non-int6) tensors
        int8_freq_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
        if int8_freq_c_len > 0:
            int8_freq_bytes = lzma.decompress(int8_block[boff:boff + int8_freq_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
        else:
            int8_freq_bytes = b""
        boff += int8_freq_c_len

        int8_tensor_bytes = []
        for ti in range(n_int8):
            numel, c_len = struct.unpack_from("<II", int8_block, boff); boff += 8
            c_bytes = int8_block[boff:boff + c_len]; boff += c_len
            freq_u16 = np.frombuffer(int8_freq_bytes[ti * 512:(ti + 1) * 512], dtype=np.uint16).astype(np.float64)
            probs = freq_u16 / freq_u16.sum()
            compressed = np.frombuffer(c_bytes, dtype=np.uint32)
            decoder = constriction.stream.stack.AnsCoder(compressed)
            model = constriction.stream.model.Categorical(probs, perfect=False)
            zigzag = decoder.decode(model, numel)
            decoded = ((zigzag.astype(np.int16) >> 1) ^ -(zigzag.astype(np.int16) & 1)).astype(np.int8)
            int8_tensor_bytes.append(decoded.tobytes())

        # Merge int6 and int8 tensors in the correct key order
        int6_idx = 0
        int8_idx = 0
        int8_raw_parts = []
        for k in header["int8_keys"]:
            base = k[:-2] if k.endswith(".q") else k
            info = header["meta"].get(base)
            if isinstance(info, dict) and info.get("type") == "int6":
                int8_raw_parts.append(int6_tensor_bytes[int6_idx])
                int6_idx += 1
            else:
                int8_raw_parts.append(int8_tensor_bytes[int8_idx])
                int8_idx += 1
        int8_raw = b"".join(int8_raw_parts)
    else:
        int8_raw = b""
    fp16_block = read_block()
    if header["fp16_keys"]:
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
        fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""
    # fp32 block is optional (omitted when empty)
    if header["fp32_keys"] and off < len(blob):
        fp32_raw_shuffled = decomp.decompress(read_block())
    else:
        fp32_raw_shuffled = b""

    # Unshuffle fp16: high bytes then low bytes → interleave
    if fp16_raw_shuffled:
        half = len(fp16_raw_shuffled) // 2
        high = np.frombuffer(fp16_raw_shuffled[:half], dtype=np.uint8)
        low = np.frombuffer(fp16_raw_shuffled[half:], dtype=np.uint8)
        interleaved = np.empty(len(fp16_raw_shuffled), dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        fp16_raw = interleaved.tobytes()
    else:
        fp16_raw = fp16_raw_shuffled

    # Unshuffle fp32: 4 sub-streams → interleave
    if fp32_raw_shuffled:
        quarter = len(fp32_raw_shuffled) // 4
        subs = [np.frombuffer(fp32_raw_shuffled[i*quarter:(i+1)*quarter], dtype=np.uint8) for i in range(4)]
        interleaved = np.empty(len(fp32_raw_shuffled), dtype=np.uint8)
        for i in range(4):
            interleaved[i::4] = subs[i]
        fp32_raw = interleaved.tobytes()
    else:
        fp32_raw = fp32_raw_shuffled

    dtype_map = {"int8": (torch.int8, int8_raw, np.int8),
                 "fp16": (torch.float16, fp16_raw, np.float16),
                 "fp32": (torch.float32, fp32_raw, np.float32)}

    w = {}
    offsets = {"int8": 0, "fp16": 0, "fp32": 0}
    # fp16 keys ordered: passthrough first, then scales (must match encoder)
    fp16_keys_ordered = sorted(header["fp16_keys"], key=lambda k: (0 if ".scale" not in k else 1, k))
    for label, keys in [("int8", header["int8_keys"]), ("fp16", fp16_keys_ordered), ("fp32", header["fp32_keys"])]:
        dt, raw, npdt = dtype_map[label]
        for k in keys:
            shape = header["shapes"][k]
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * np.dtype(npdt).itemsize
            arr = np.frombuffer(raw, dtype=npdt, count=numel, offset=offsets[label])
            offsets[label] += nbytes
            t = torch.from_numpy(arr.copy()).reshape(shape)
            w[k] = t
    return w, header["meta"]


# ==============================================================================
# EXPERIMENT: Brotli-11 variants (inspired by PR #1508)
# ==============================================================================

def _best_compress(data: bytes, candidates: dict[str, bytes]) -> tuple[str, bytes]:
    """Return the (name, blob) pair with the smallest size."""
    best_name = min(candidates, key=lambda k: len(candidates[k]))
    return best_name, candidates[best_name]


def encode_exp63_brotli_fp16(quant_result: dict[str, Tensor], quant_meta: dict[str, object],
                              arch_params: dict | None = None) -> bytes:
    """exp63: Try brotli-11 for the fp16 stream instead of LZMA.

    Hypothesis: Brotli-11's LZ77+Huffman may beat LZMA on the byte-shuffled
    fp16 stream since fp16 exponent bytes are highly repetitive, and brotli
    is known to excel on structured repetitive data (per PR #1508 findings).
    """
    if not HAS_BROTLI:
        raise ImportError("brotli not installed — pip install brotli")
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    comp = zstandard.ZstdCompressor(level=22)

    # Reuse the experiment encoder but swap fp16 compressor
    # We do this by calling the experiment encoder internals
    return _encode_with_fp16_compressor(quant_result, quant_meta, arch_params, fp16_compressor="brotli")


def decode_exp63_brotli_fp16(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """exp63 decoder: detect brotli vs LZMA for fp16 block."""
    return _decode_with_fp16_compressor(blob, fp16_compressor="brotli")


def encode_exp64_brotli_fp32(quant_result: dict[str, Tensor], quant_meta: dict[str, object],
                              arch_params: dict | None = None) -> bytes:
    """exp64: Try brotli-11 for the fp32 stream instead of zstd-22."""
    if not HAS_BROTLI:
        raise ImportError("brotli not installed — pip install brotli")
    return _encode_with_fp32_compressor(quant_result, quant_meta, arch_params, fp32_compressor="brotli")


def decode_exp64_brotli_fp32(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """exp64 decoder."""
    return _decode_with_fp32_compressor(blob, fp32_compressor="brotli")


def encode_exp65_brotli_best(quant_result: dict[str, Tensor], quant_meta: dict[str, object],
                              arch_params: dict | None = None) -> bytes:
    """exp65: Per-stream best-of brotli-11 vs current compressor.

    For each stream, try both the current compressor and brotli-11,
    pick whichever is smaller. A 1-byte flag per stream indicates which was used.
    """
    if not HAS_BROTLI:
        raise ImportError("brotli not installed — pip install brotli")
    return _encode_best_of(quant_result, quant_meta, arch_params)


def decode_exp65_brotli_best(blob: bytes) -> tuple[dict[str, Tensor], dict[str, object]]:
    """exp65 decoder: reads per-stream compressor flags."""
    return _decode_best_of(blob)


def _encode_with_fp16_compressor(quant_result, quant_meta, arch_params, fp16_compressor="lzma"):
    """Variant of encode_experiment that allows swapping the fp16 compressor."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    comp = zstandard.ZstdCompressor(level=22)

    # Classify tensors
    int8_keys, fp16_keys, fp32_keys = [], [], []
    for k in _sorted_keys(quant_result):
        t = quant_result[k]
        if t.dtype == torch.int8: int8_keys.append(k)
        elif t.dtype == torch.float16: fp16_keys.append(k)
        else: fp32_keys.append(k)

    streams = {}
    import constriction
    from scipy.cluster.vq import kmeans2

    # === INT6 + INT8 ANS (identical to encode_experiment) ===
    int6_keys, int8_only_keys = [], []
    for k in int8_keys:
        base = k[:-2] if k.endswith(".q") else k
        info = quant_meta.get(base)
        if isinstance(info, dict) and info.get("type") == "int6":
            int6_keys.append(k)
        else:
            int8_only_keys.append(k)

    N_CLUSTERS = 16
    int6_row_data, int6_row_dists = [], []
    int6_row_tensor_idx, int6_tensor_nrows = [], []

    for ti, k in enumerate(int6_keys):
        t = quant_result[k].contiguous().numpy()
        if t.ndim != 2:
            arr = t.astype(np.int16).flatten()
            zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_row_tensor_idx.append(ti)
            int6_tensor_nrows.append(1)
            continue
        nrows = t.shape[0]
        int6_tensor_nrows.append(nrows)
        for i in range(nrows):
            row = t[i].astype(np.int16)
            zigzag = ((row << 1) ^ (row >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_row_tensor_idx.append(ti)

    dist_matrix = np.sqrt(np.array(int6_row_dists))
    centroids, labels = kmeans2(dist_matrix, N_CLUSTERS, minit="random", iter=100, seed=286)

    cluster_freq_u16, cluster_models = [], []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if not mask.any():
            cluster_freq_u16.append(np.ones(64, dtype=np.uint16))
            cluster_models.append(None)
            continue
        cluster_data = np.concatenate([int6_row_data[i] for i in range(len(labels)) if labels[i] == c])
        freqs = np.bincount(cluster_data, minlength=64)[:64].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        cluster_freq_u16.append(freq_u16)
        cluster_models.append(constriction.stream.model.Categorical(probs, perfect=False))

    encoder = constriction.stream.stack.AnsCoder()
    for i in range(len(int6_row_data) - 1, -1, -1):
        encoder.encode_reverse(int6_row_data[i], cluster_models[labels[i]])
    int6_compressed = encoder.get_compressed().tobytes()

    if N_CLUSTERS <= 16:
        packed_labels = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
        for i in range(0, len(labels) - 1, 2):
            packed_labels[i // 2] = (labels[i] << 4) | labels[i + 1]
        if len(labels) % 2:
            packed_labels[-1] = labels[-1] << 4
    else:
        packed_labels = labels.astype(np.uint8)
    freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    labels_compressed = lzma.compress(packed_labels.tobytes(), format=lzma.FORMAT_RAW, filters=freq_filters)
    all_freq_bytes = b"".join(f.tobytes() for f in cluster_freq_u16)
    freq_compressed = lzma.compress(all_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters)

    int8_parts = []
    for k in int8_only_keys:
        t = quant_result[k].contiguous()
        arr = t.numpy().astype(np.int16).flatten()
        zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
        freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        enc = constriction.stream.stack.AnsCoder()
        enc.encode_reverse(zigzag, model)
        int8_parts.append((enc.get_compressed().tobytes(), freq_u16, len(zigzag)))

    ans_out = struct.pack("<BBHHI", 1, N_CLUSTERS, len(int6_keys), len(int8_only_keys), len(labels))
    ans_out += struct.pack("<H", len(freq_compressed)) + freq_compressed
    ans_out += struct.pack("<H", len(labels_compressed)) + labels_compressed
    ans_out += struct.pack("<I", len(int6_compressed)) + int6_compressed
    int8_freq_bytes = b"".join(f.tobytes() for _, f, _ in int8_parts)
    int8_freq_c = lzma.compress(int8_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters) if int8_parts else b""
    ans_out += struct.pack("<H", len(int8_freq_c)) + int8_freq_c
    for c_data_bytes, _, numel in int8_parts:
        ans_out += struct.pack("<II", numel, len(c_data_bytes)) + c_data_bytes
    streams["int8"] = ans_out

    # === FP16: byte-shuffle + configurable compressor ===
    fp16_ordered = sorted(fp16_keys, key=lambda k: (0 if ".scale" not in k else 1, k))
    fp16_parts = []
    for k in fp16_ordered:
        t = quant_result[k].contiguous()
        fp16_parts.append(t.numpy().tobytes())
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        high = arr[0::2].tobytes()
        low = arr[1::2].tobytes()
        shuffled = high + low
        if fp16_compressor == "brotli" and HAS_BROTLI:
            streams["fp16"] = _brotli.compress(shuffled, quality=11)
        else:
            filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
            streams["fp16"] = lzma.compress(shuffled, format=lzma.FORMAT_RAW, filters=filters)

    # === FP32: byte-shuffle + zstd-22 ===
    fp32_parts = []
    for k in fp32_keys:
        t = quant_result[k]
        if t.ndim == 2:
            t = t.t().contiguous()
        fp32_parts.append(t.numpy().tobytes())
    if fp32_parts:
        raw = b"".join(fp32_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = b"".join(arr[i::4].tobytes() for i in range(4))
        streams["fp32"] = comp.compress(shuffled)

    # === Header ===
    all_keys = _sorted_keys(quant_result)
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set:
            pass
        elif k not in meta_keys:
            meta_keys.append(k)
    type_str = ""
    for mk in meta_keys:
        info = quant_meta.get(mk, "passthrough")
        if info == "passthrough": type_str += "p"
        elif info == "passthrough_ctrl": type_str += "c"
        elif isinstance(info, dict) and info.get("type") == "int6": type_str += "6"
        elif isinstance(info, dict) and info.get("type") == "int8": type_str += "8"
        else: type_str += "p"
    dtype_str = ""
    for k in all_keys:
        t = quant_result[k]
        if t.dtype == torch.int8: dtype_str += "i"
        elif t.dtype == torch.float16: dtype_str += "h"
        else: dtype_str += "f"

    if arch_params is None:
        arch_params = {"D": 512, "H": 8, "KV": 4, "MLP": 3, "L": 11,
                       "vocab": 1024, "bigram_dim": 128}
    header = json.dumps({"keys": all_keys, "arch": arch_params,
                         "types": type_str, "dtypes": dtype_str},
                        separators=(",", ":")).encode()
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_c = lzma.compress(header, format=lzma.FORMAT_RAW, filters=header_filters)

    out = struct.pack("<H", len(header_c)) + header_c
    for label in ["int8", "fp16"]:
        blob = streams.get(label, b"")
        out += struct.pack("<I", len(blob)) + blob
    fp32_blob = streams.get("fp32", b"")
    if fp32_blob:
        out += struct.pack("<I", len(fp32_blob)) + fp32_blob
    return out


def _decode_with_fp16_compressor(blob, fp16_compressor="lzma"):
    """Variant of decode_experiment that uses brotli for the fp16 block."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    decomp = zstandard.ZstdDecompressor()

    off = 0
    def read_block():
        nonlocal off
        sz = struct.unpack_from("<I", blob, off)[0]; off_val = off + 4
        data = blob[off_val:off_val + sz]
        # update nonlocal
        return sz, data

    # Header
    header_sz = struct.unpack_from("<H", blob, off)[0]; off += 2
    raw_header = blob[off:off + header_sz]; off += header_sz
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_json = json.loads(lzma.decompress(raw_header, format=lzma.FORMAT_RAW, filters=header_filters))

    all_keys = header_json["keys"]
    arch = header_json["arch"]
    if isinstance(arch, list):
        arch = {"D": arch[0], "H": arch[1], "KV": arch[2], "MLP": arch[3], "L": arch[4],
                "vocab": arch[5] if len(arch) > 5 else 1024, "bigram_dim": arch[6] if len(arch) > 6 else 128}
    shapes_list = [_derive_shape(k, arch) for k in all_keys]

    dtype_str = header_json.get("dtypes", "")
    int8_indices = [i for i, c in enumerate(dtype_str) if c == "i"]
    fp16_indices = [i for i, c in enumerate(dtype_str) if c == "h"]
    fp32_indices = [i for i, c in enumerate(dtype_str) if c == "f"]

    type_str = header_json.get("types", "")
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set: pass
        elif k not in meta_keys: meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta_decoded = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}

    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "meta": quant_meta_decoded,
    }

    # int8 block (ANS) — use the original decoder
    import constriction
    int8_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    int8_block = blob[off:off + int8_sz]; off += int8_sz
    int8_raw = _decode_ans_block(int8_block, header) if int8_block else b""

    # fp16 block
    fp16_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    fp16_block = blob[off:off + fp16_sz]; off += fp16_sz
    if header["fp16_keys"] and fp16_block:
        if fp16_compressor == "brotli" and HAS_BROTLI:
            fp16_raw_shuffled = _brotli.decompress(fp16_block)
        else:
            filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
            fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""

    # fp32 block
    if header["fp32_keys"] and off < len(blob):
        fp32_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
        fp32_raw_shuffled = decomp.decompress(blob[off:off + fp32_sz])
    else:
        fp32_raw_shuffled = b""

    # Unshuffle and reconstruct
    return _reconstruct_tensors(header, int8_raw, fp16_raw_shuffled, fp32_raw_shuffled)


def _encode_with_fp32_compressor(quant_result, quant_meta, arch_params, fp32_compressor="zstd"):
    """Variant of encode_experiment that allows swapping the fp32 compressor."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    comp = zstandard.ZstdCompressor(level=22)

    int8_keys, fp16_keys, fp32_keys = [], [], []
    for k in _sorted_keys(quant_result):
        t = quant_result[k]
        if t.dtype == torch.int8: int8_keys.append(k)
        elif t.dtype == torch.float16: fp16_keys.append(k)
        else: fp32_keys.append(k)

    streams = {}
    import constriction
    from scipy.cluster.vq import kmeans2

    # INT6 + INT8 ANS (same as experiment)
    int6_keys, int8_only_keys = [], []
    for k in int8_keys:
        base = k[:-2] if k.endswith(".q") else k
        info = quant_meta.get(base)
        if isinstance(info, dict) and info.get("type") == "int6": int6_keys.append(k)
        else: int8_only_keys.append(k)

    N_CLUSTERS = 16
    int6_row_data, int6_row_dists, int6_tensor_nrows = [], [], []
    for ti, k in enumerate(int6_keys):
        t = quant_result[k].contiguous().numpy()
        if t.ndim != 2:
            arr = t.astype(np.int16).flatten()
            zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_tensor_nrows.append(1)
            continue
        int6_tensor_nrows.append(t.shape[0])
        for i in range(t.shape[0]):
            row = t[i].astype(np.int16)
            zigzag = ((row << 1) ^ (row >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())

    dist_matrix = np.sqrt(np.array(int6_row_dists))
    centroids, labels = kmeans2(dist_matrix, N_CLUSTERS, minit="random", iter=100, seed=286)
    cluster_freq_u16, cluster_models = [], []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if not mask.any():
            cluster_freq_u16.append(np.ones(64, dtype=np.uint16))
            cluster_models.append(None)
            continue
        cluster_data = np.concatenate([int6_row_data[i] for i in range(len(labels)) if labels[i] == c])
        freqs = np.bincount(cluster_data, minlength=64)[:64].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        cluster_freq_u16.append(freq_u16)
        cluster_models.append(constriction.stream.model.Categorical(probs, perfect=False))

    encoder = constriction.stream.stack.AnsCoder()
    for i in range(len(int6_row_data) - 1, -1, -1):
        encoder.encode_reverse(int6_row_data[i], cluster_models[labels[i]])
    int6_compressed = encoder.get_compressed().tobytes()
    if N_CLUSTERS <= 16:
        packed_labels = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
        for i in range(0, len(labels) - 1, 2):
            packed_labels[i // 2] = (labels[i] << 4) | labels[i + 1]
        if len(labels) % 2: packed_labels[-1] = labels[-1] << 4
    else:
        packed_labels = labels.astype(np.uint8)
    freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    labels_compressed = lzma.compress(packed_labels.tobytes(), format=lzma.FORMAT_RAW, filters=freq_filters)
    all_freq_bytes = b"".join(f.tobytes() for f in cluster_freq_u16)
    freq_compressed = lzma.compress(all_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters)
    int8_parts = []
    for k in int8_only_keys:
        t = quant_result[k].contiguous()
        arr = t.numpy().astype(np.int16).flatten()
        zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
        freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        enc = constriction.stream.stack.AnsCoder()
        enc.encode_reverse(zigzag, model)
        int8_parts.append((enc.get_compressed().tobytes(), freq_u16, len(zigzag)))
    ans_out = struct.pack("<BBHHI", 1, N_CLUSTERS, len(int6_keys), len(int8_only_keys), len(labels))
    ans_out += struct.pack("<H", len(freq_compressed)) + freq_compressed
    ans_out += struct.pack("<H", len(labels_compressed)) + labels_compressed
    ans_out += struct.pack("<I", len(int6_compressed)) + int6_compressed
    int8_freq_bytes = b"".join(f.tobytes() for _, f, _ in int8_parts)
    int8_freq_c = lzma.compress(int8_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters) if int8_parts else b""
    ans_out += struct.pack("<H", len(int8_freq_c)) + int8_freq_c
    for c_data_bytes, _, numel in int8_parts:
        ans_out += struct.pack("<II", numel, len(c_data_bytes)) + c_data_bytes
    streams["int8"] = ans_out

    # fp16: LZMA (unchanged)
    fp16_ordered = sorted(fp16_keys, key=lambda k: (0 if ".scale" not in k else 1, k))
    fp16_parts = [quant_result[k].contiguous().numpy().tobytes() for k in fp16_ordered]
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = arr[0::2].tobytes() + arr[1::2].tobytes()
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
        streams["fp16"] = lzma.compress(shuffled, format=lzma.FORMAT_RAW, filters=filters)

    # fp32: configurable compressor
    fp32_parts = []
    for k in fp32_keys:
        t = quant_result[k]
        if t.ndim == 2: t = t.t().contiguous()
        fp32_parts.append(t.numpy().tobytes())
    if fp32_parts:
        raw = b"".join(fp32_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = b"".join(arr[i::4].tobytes() for i in range(4))
        if fp32_compressor == "brotli" and HAS_BROTLI:
            streams["fp32"] = _brotli.compress(shuffled, quality=11)
        else:
            streams["fp32"] = comp.compress(shuffled)

    # Header (same as experiment)
    all_keys = _sorted_keys(quant_result)
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set: pass
        elif k not in meta_keys: meta_keys.append(k)
    type_str = ""
    for mk in meta_keys:
        info = quant_meta.get(mk, "passthrough")
        if info == "passthrough": type_str += "p"
        elif info == "passthrough_ctrl": type_str += "c"
        elif isinstance(info, dict) and info.get("type") == "int6": type_str += "6"
        elif isinstance(info, dict) and info.get("type") == "int8": type_str += "8"
        else: type_str += "p"
    dtype_str = ""
    for k in all_keys:
        t = quant_result[k]
        if t.dtype == torch.int8: dtype_str += "i"
        elif t.dtype == torch.float16: dtype_str += "h"
        else: dtype_str += "f"
    if arch_params is None:
        arch_params = {"D": 512, "H": 8, "KV": 4, "MLP": 3, "L": 11,
                       "vocab": 1024, "bigram_dim": 128}
    hdr = json.dumps({"keys": all_keys, "arch": arch_params,
                       "types": type_str, "dtypes": dtype_str},
                      separators=(",", ":")).encode()
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_c = lzma.compress(hdr, format=lzma.FORMAT_RAW, filters=header_filters)
    out = struct.pack("<H", len(header_c)) + header_c
    for label in ["int8", "fp16"]:
        b = streams.get(label, b"")
        out += struct.pack("<I", len(b)) + b
    fp32_blob = streams.get("fp32", b"")
    if fp32_blob:
        out += struct.pack("<I", len(fp32_blob)) + fp32_blob
    return out


def _decode_with_fp32_compressor(blob, fp32_compressor="zstd"):
    """Variant of decode_experiment that uses brotli for the fp32 block."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    decomp = zstandard.ZstdDecompressor()

    off = 0
    header_sz = struct.unpack_from("<H", blob, off)[0]; off += 2
    raw_header = blob[off:off + header_sz]; off += header_sz
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_json = json.loads(lzma.decompress(raw_header, format=lzma.FORMAT_RAW, filters=header_filters))

    all_keys = header_json["keys"]
    arch = header_json["arch"]
    if isinstance(arch, list):
        arch = {"D": arch[0], "H": arch[1], "KV": arch[2], "MLP": arch[3], "L": arch[4],
                "vocab": arch[5] if len(arch) > 5 else 1024, "bigram_dim": arch[6] if len(arch) > 6 else 128}
    shapes_list = [_derive_shape(k, arch) for k in all_keys]
    dtype_str = header_json.get("dtypes", "")
    int8_indices = [i for i, c in enumerate(dtype_str) if c == "i"]
    fp16_indices = [i for i, c in enumerate(dtype_str) if c == "h"]
    fp32_indices = [i for i, c in enumerate(dtype_str) if c == "f"]
    type_str = header_json.get("types", "")
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set: pass
        elif k not in meta_keys: meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta_decoded = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}
    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "meta": quant_meta_decoded,
    }

    import constriction
    int8_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    int8_block = blob[off:off + int8_sz]; off += int8_sz
    int8_raw = _decode_ans_block(int8_block, header) if int8_block else b""

    fp16_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    fp16_block = blob[off:off + fp16_sz]; off += fp16_sz
    if header["fp16_keys"] and fp16_block:
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
        fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""

    if header["fp32_keys"] and off < len(blob):
        fp32_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
        fp32_data = blob[off:off + fp32_sz]
        if fp32_compressor == "brotli" and HAS_BROTLI:
            fp32_raw_shuffled = _brotli.decompress(fp32_data)
        else:
            fp32_raw_shuffled = decomp.decompress(fp32_data)
    else:
        fp32_raw_shuffled = b""

    return _reconstruct_tensors(header, int8_raw, fp16_raw_shuffled, fp32_raw_shuffled)


def _encode_best_of(quant_result, quant_meta, arch_params):
    """exp65: Per-stream best-of brotli-11 vs current compressor.

    Format: identical to experiment, but with a 1-byte flags field after
    the header indicating which compressor was used per stream.
    Bit 0: fp16 (0=LZMA, 1=brotli)
    Bit 1: fp32 (0=zstd, 1=brotli)
    """
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    if not HAS_BROTLI:
        raise ImportError("brotli not installed")
    comp = zstandard.ZstdCompressor(level=22)

    int8_keys, fp16_keys, fp32_keys = [], [], []
    for k in _sorted_keys(quant_result):
        t = quant_result[k]
        if t.dtype == torch.int8: int8_keys.append(k)
        elif t.dtype == torch.float16: fp16_keys.append(k)
        else: fp32_keys.append(k)

    streams = {}
    import constriction
    from scipy.cluster.vq import kmeans2

    # INT6 + INT8 ANS (same as experiment — no brotli alternative for ANS)
    int6_keys, int8_only_keys = [], []
    for k in int8_keys:
        base = k[:-2] if k.endswith(".q") else k
        info = quant_meta.get(base)
        if isinstance(info, dict) and info.get("type") == "int6": int6_keys.append(k)
        else: int8_only_keys.append(k)

    N_CLUSTERS = 16
    int6_row_data, int6_row_dists, int6_tensor_nrows = [], [], []
    for ti, k in enumerate(int6_keys):
        t = quant_result[k].contiguous().numpy()
        if t.ndim != 2:
            arr = t.astype(np.int16).flatten()
            zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())
            int6_tensor_nrows.append(1)
            continue
        int6_tensor_nrows.append(t.shape[0])
        for i in range(t.shape[0]):
            row = t[i].astype(np.int16)
            zigzag = ((row << 1) ^ (row >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            freqs = np.maximum(freqs, 1)
            int6_row_data.append(zigzag)
            int6_row_dists.append(freqs / freqs.sum())

    dist_matrix = np.sqrt(np.array(int6_row_dists))
    centroids, labels = kmeans2(dist_matrix, N_CLUSTERS, minit="random", iter=100, seed=286)
    cluster_freq_u16, cluster_models = [], []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if not mask.any():
            cluster_freq_u16.append(np.ones(64, dtype=np.uint16))
            cluster_models.append(None)
            continue
        cluster_data = np.concatenate([int6_row_data[i] for i in range(len(labels)) if labels[i] == c])
        freqs = np.bincount(cluster_data, minlength=64)[:64].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        cluster_freq_u16.append(freq_u16)
        cluster_models.append(constriction.stream.model.Categorical(probs, perfect=False))

    encoder = constriction.stream.stack.AnsCoder()
    for i in range(len(int6_row_data) - 1, -1, -1):
        encoder.encode_reverse(int6_row_data[i], cluster_models[labels[i]])
    int6_compressed = encoder.get_compressed().tobytes()
    if N_CLUSTERS <= 16:
        packed_labels = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
        for i in range(0, len(labels) - 1, 2):
            packed_labels[i // 2] = (labels[i] << 4) | labels[i + 1]
        if len(labels) % 2: packed_labels[-1] = labels[-1] << 4
    else:
        packed_labels = labels.astype(np.uint8)
    freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    labels_compressed = lzma.compress(packed_labels.tobytes(), format=lzma.FORMAT_RAW, filters=freq_filters)
    all_freq_bytes = b"".join(f.tobytes() for f in cluster_freq_u16)
    freq_compressed = lzma.compress(all_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters)
    int8_parts = []
    for k in int8_only_keys:
        t = quant_result[k].contiguous()
        arr = t.numpy().astype(np.int16).flatten()
        zigzag = ((arr << 1) ^ (arr >> 15)).astype(np.int32)
        freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        model = constriction.stream.model.Categorical(probs, perfect=False)
        enc = constriction.stream.stack.AnsCoder()
        enc.encode_reverse(zigzag, model)
        int8_parts.append((enc.get_compressed().tobytes(), freq_u16, len(zigzag)))
    ans_out = struct.pack("<BBHHI", 1, N_CLUSTERS, len(int6_keys), len(int8_only_keys), len(labels))
    ans_out += struct.pack("<H", len(freq_compressed)) + freq_compressed
    ans_out += struct.pack("<H", len(labels_compressed)) + labels_compressed
    ans_out += struct.pack("<I", len(int6_compressed)) + int6_compressed
    int8_freq_bytes = b"".join(f.tobytes() for _, f, _ in int8_parts)
    int8_freq_c = lzma.compress(int8_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters) if int8_parts else b""
    ans_out += struct.pack("<H", len(int8_freq_c)) + int8_freq_c
    for c_data_bytes, _, numel in int8_parts:
        ans_out += struct.pack("<II", numel, len(c_data_bytes)) + c_data_bytes
    streams["int8"] = ans_out

    # fp16: try both LZMA and brotli, pick smaller
    flags = 0
    fp16_ordered = sorted(fp16_keys, key=lambda k: (0 if ".scale" not in k else 1, k))
    fp16_parts = [quant_result[k].contiguous().numpy().tobytes() for k in fp16_ordered]
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = arr[0::2].tobytes() + arr[1::2].tobytes()
        lzma_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
        fp16_lzma = lzma.compress(shuffled, format=lzma.FORMAT_RAW, filters=lzma_filters)
        fp16_brotli = _brotli.compress(shuffled, quality=11)
        if len(fp16_brotli) < len(fp16_lzma):
            streams["fp16"] = fp16_brotli
            flags |= 1
        else:
            streams["fp16"] = fp16_lzma

    # fp32: try both zstd and brotli, pick smaller
    fp32_parts = []
    for k in fp32_keys:
        t = quant_result[k]
        if t.ndim == 2: t = t.t().contiguous()
        fp32_parts.append(t.numpy().tobytes())
    if fp32_parts:
        raw = b"".join(fp32_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        shuffled = b"".join(arr[i::4].tobytes() for i in range(4))
        fp32_zstd = comp.compress(shuffled)
        fp32_brotli = _brotli.compress(shuffled, quality=11)
        if len(fp32_brotli) < len(fp32_zstd):
            streams["fp32"] = fp32_brotli
            flags |= 2
        else:
            streams["fp32"] = fp32_zstd

    # Header
    all_keys = _sorted_keys(quant_result)
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set: pass
        elif k not in meta_keys: meta_keys.append(k)
    type_str = ""
    for mk in meta_keys:
        info = quant_meta.get(mk, "passthrough")
        if info == "passthrough": type_str += "p"
        elif info == "passthrough_ctrl": type_str += "c"
        elif isinstance(info, dict) and info.get("type") == "int6": type_str += "6"
        elif isinstance(info, dict) and info.get("type") == "int8": type_str += "8"
        else: type_str += "p"
    dtype_str = ""
    for k in all_keys:
        t = quant_result[k]
        if t.dtype == torch.int8: dtype_str += "i"
        elif t.dtype == torch.float16: dtype_str += "h"
        else: dtype_str += "f"
    if arch_params is None:
        arch_params = {"D": 512, "H": 8, "KV": 4, "MLP": 3, "L": 11,
                       "vocab": 1024, "bigram_dim": 128}
    hdr = json.dumps({"keys": all_keys, "arch": arch_params,
                       "types": type_str, "dtypes": dtype_str},
                      separators=(",", ":")).encode()
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_c = lzma.compress(hdr, format=lzma.FORMAT_RAW, filters=header_filters)

    # Output with flags byte after header
    out = struct.pack("<H", len(header_c)) + header_c
    out += struct.pack("<B", flags)  # 1 byte compressor flags
    for label in ["int8", "fp16"]:
        b = streams.get(label, b"")
        out += struct.pack("<I", len(b)) + b
    fp32_blob = streams.get("fp32", b"")
    if fp32_blob:
        out += struct.pack("<I", len(fp32_blob)) + fp32_blob
    return out


def _decode_best_of(blob):
    """exp65 decoder: reads per-stream compressor flags."""
    if not HAS_ZSTD:
        raise ImportError("zstandard not installed")
    decomp = zstandard.ZstdDecompressor()

    off = 0
    header_sz = struct.unpack_from("<H", blob, off)[0]; off += 2
    raw_header = blob[off:off + header_sz]; off += header_sz
    header_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
    header_json = json.loads(lzma.decompress(raw_header, format=lzma.FORMAT_RAW, filters=header_filters))

    all_keys = header_json["keys"]
    arch = header_json["arch"]
    if isinstance(arch, list):
        arch = {"D": arch[0], "H": arch[1], "KV": arch[2], "MLP": arch[3], "L": arch[4],
                "vocab": arch[5] if len(arch) > 5 else 1024, "bigram_dim": arch[6] if len(arch) > 6 else 128}
    shapes_list = [_derive_shape(k, arch) for k in all_keys]
    dtype_str = header_json.get("dtypes", "")
    int8_indices = [i for i, c in enumerate(dtype_str) if c == "i"]
    fp16_indices = [i for i, c in enumerate(dtype_str) if c == "h"]
    fp32_indices = [i for i, c in enumerate(dtype_str) if c == "f"]
    type_str = header_json.get("types", "")
    all_keys_set = set(all_keys)
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q"):
            base = k[:-2]
            if base not in meta_keys: meta_keys.append(base)
        elif k.endswith(".scale") and (k[:-6] + ".q") in all_keys_set: pass
        elif k not in meta_keys: meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta_decoded = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}
    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "meta": quant_meta_decoded,
    }

    # Read flags
    flags = struct.unpack_from("<B", blob, off)[0]; off += 1
    fp16_is_brotli = bool(flags & 1)
    fp32_is_brotli = bool(flags & 2)

    import constriction
    int8_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    int8_block = blob[off:off + int8_sz]; off += int8_sz
    int8_raw = _decode_ans_block(int8_block, header) if int8_block else b""

    fp16_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
    fp16_block = blob[off:off + fp16_sz]; off += fp16_sz
    if header["fp16_keys"] and fp16_block:
        if fp16_is_brotli and HAS_BROTLI:
            fp16_raw_shuffled = _brotli.decompress(fp16_block)
        else:
            filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]
            fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""

    if header["fp32_keys"] and off < len(blob):
        fp32_sz = struct.unpack_from("<I", blob, off)[0]; off += 4
        fp32_data = blob[off:off + fp32_sz]
        if fp32_is_brotli and HAS_BROTLI:
            fp32_raw_shuffled = _brotli.decompress(fp32_data)
        else:
            fp32_raw_shuffled = decomp.decompress(fp32_data)
    else:
        fp32_raw_shuffled = b""

    return _reconstruct_tensors(header, int8_raw, fp16_raw_shuffled, fp32_raw_shuffled)


def _decode_ans_block(int8_block, header):
    """Shared ANS decoding logic for the int8 block (extracted from decode_experiment)."""
    import constriction

    boff = 0
    version, n_clusters, n_int6, n_int8, n_rows = struct.unpack_from("<BBHHI", int8_block, boff)
    boff += 10
    freq_filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 0, "pb": 0}]

    freq_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
    all_freq_bytes = lzma.decompress(int8_block[boff:boff + freq_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
    boff += freq_c_len

    labels_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
    labels_packed = lzma.decompress(int8_block[boff:boff + labels_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
    boff += labels_c_len
    labels_arr = np.frombuffer(labels_packed, dtype=np.uint8)
    if n_clusters <= 16:
        labels = np.empty(n_rows, dtype=np.int32)
        for i in range(0, n_rows - 1, 2):
            labels[i] = labels_arr[i // 2] >> 4
            labels[i + 1] = labels_arr[i // 2] & 0x0F
        if n_rows % 2:
            labels[n_rows - 1] = labels_arr[n_rows // 2] >> 4
    else:
        labels = labels_arr[:n_rows].astype(np.int32)

    cluster_probs = []
    for c in range(n_clusters):
        freq_u16 = np.frombuffer(all_freq_bytes[c * 128:(c + 1) * 128], dtype=np.uint16).astype(np.float64)
        probs = freq_u16 / freq_u16.sum()
        cluster_probs.append(probs)

    int6_c_len = struct.unpack_from("<I", int8_block, boff)[0]; boff += 4
    int6_c_bytes = int8_block[boff:boff + int6_c_len]; boff += int6_c_len

    int6_q_keys = [k for k in header["int8_keys"]
                   if isinstance(header["meta"].get(k[:-2] if k.endswith(".q") else k), dict)
                   and header["meta"].get(k[:-2] if k.endswith(".q") else k, {}).get("type") == "int6"]
    tensor_nrows = []
    for k in int6_q_keys:
        shape = header["shapes"][k]
        tensor_nrows.append(shape[0] if len(shape) >= 2 else 1)

    row_ncols = []
    for ti, nr in enumerate(tensor_nrows):
        shape = header["shapes"][int6_q_keys[ti]]
        ncols = shape[-1] if len(shape) >= 2 else int(np.prod(shape))
        for _ in range(nr):
            row_ncols.append(ncols)

    cluster_models_dec = [constriction.stream.model.Categorical(p, perfect=False) for p in cluster_probs]
    int6_decoder = constriction.stream.stack.AnsCoder(np.frombuffer(int6_c_bytes, dtype=np.uint32))
    int6_decoded = []
    for row_idx in range(n_rows):
        c = int(labels[row_idx])
        ncols = row_ncols[row_idx]
        row_zigzag = int6_decoder.decode(cluster_models_dec[c], ncols)
        decoded = ((row_zigzag.astype(np.int16) >> 1) ^ -(row_zigzag.astype(np.int16) & 1)).astype(np.int8)
        int6_decoded.append(decoded.tobytes())

    int6_tensor_bytes = []
    row_cursor = 0
    for nr in tensor_nrows:
        int6_tensor_bytes.append(b"".join(int6_decoded[row_cursor:row_cursor + nr]))
        row_cursor += nr

    int8_freq_c_len = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
    if int8_freq_c_len > 0:
        int8_freq_bytes = lzma.decompress(int8_block[boff:boff + int8_freq_c_len], format=lzma.FORMAT_RAW, filters=freq_filters)
    else:
        int8_freq_bytes = b""
    boff += int8_freq_c_len

    int8_tensor_bytes = []
    for ti in range(n_int8):
        numel, c_len = struct.unpack_from("<II", int8_block, boff); boff += 8
        c_bytes = int8_block[boff:boff + c_len]; boff += c_len
        freq_u16 = np.frombuffer(int8_freq_bytes[ti * 512:(ti + 1) * 512], dtype=np.uint16).astype(np.float64)
        probs = freq_u16 / freq_u16.sum()
        compressed = np.frombuffer(c_bytes, dtype=np.uint32)
        decoder = constriction.stream.stack.AnsCoder(compressed)
        model = constriction.stream.model.Categorical(probs, perfect=False)
        zigzag = decoder.decode(model, numel)
        decoded = ((zigzag.astype(np.int16) >> 1) ^ -(zigzag.astype(np.int16) & 1)).astype(np.int8)
        int8_tensor_bytes.append(decoded.tobytes())

    int6_idx, int8_idx = 0, 0
    int8_raw_parts = []
    for k in header["int8_keys"]:
        base = k[:-2] if k.endswith(".q") else k
        info = header["meta"].get(base)
        if isinstance(info, dict) and info.get("type") == "int6":
            int8_raw_parts.append(int6_tensor_bytes[int6_idx]); int6_idx += 1
        else:
            int8_raw_parts.append(int8_tensor_bytes[int8_idx]); int8_idx += 1
    return b"".join(int8_raw_parts)


def _reconstruct_tensors(header, int8_raw, fp16_raw_shuffled, fp32_raw_shuffled):
    """Shared tensor reconstruction from raw byte streams."""
    # Unshuffle fp16
    if fp16_raw_shuffled:
        half = len(fp16_raw_shuffled) // 2
        high = np.frombuffer(fp16_raw_shuffled[:half], dtype=np.uint8)
        low = np.frombuffer(fp16_raw_shuffled[half:], dtype=np.uint8)
        interleaved = np.empty(len(fp16_raw_shuffled), dtype=np.uint8)
        interleaved[0::2] = high
        interleaved[1::2] = low
        fp16_raw = interleaved.tobytes()
    else:
        fp16_raw = b""

    # Unshuffle fp32
    if fp32_raw_shuffled:
        quarter = len(fp32_raw_shuffled) // 4
        subs = [np.frombuffer(fp32_raw_shuffled[i*quarter:(i+1)*quarter], dtype=np.uint8) for i in range(4)]
        interleaved = np.empty(len(fp32_raw_shuffled), dtype=np.uint8)
        for i in range(4):
            interleaved[i::4] = subs[i]
        fp32_raw = interleaved.tobytes()
    else:
        fp32_raw = b""

    dtype_map = {"int8": (torch.int8, int8_raw, np.int8),
                 "fp16": (torch.float16, fp16_raw, np.float16),
                 "fp32": (torch.float32, fp32_raw, np.float32)}

    w = {}
    offsets = {"int8": 0, "fp16": 0, "fp32": 0}
    fp16_keys_ordered = sorted(header["fp16_keys"], key=lambda k: (0 if ".scale" not in k else 1, k))
    for label, keys in [("int8", header["int8_keys"]), ("fp16", fp16_keys_ordered), ("fp32", header["fp32_keys"])]:
        dt, raw, npdt = dtype_map[label]
        for k in keys:
            shape = header["shapes"][k]
            numel = 1
            for s in shape: numel *= s
            nbytes = numel * np.dtype(npdt).itemsize
            arr = np.frombuffer(raw, dtype=npdt, count=numel, offset=offsets[label])
            offsets[label] += nbytes
            w[k] = torch.from_numpy(arr.copy()).reshape(shape)
    return w, header["meta"]


# ==============================================================================
# EXPERIMENT: Entropy regularization simulation (inspired by PR #1592)
# ==============================================================================

def entropy_regularization_diagnostic(quant_result: dict[str, Tensor],
                                       quant_meta: dict[str, object]) -> dict:
    """exp66: Simulate entropy regularization's effect on compressibility.

    PR #1592 proposed adding L2 weight decay ramp during warmdown to push
    weights toward zero, making post-quantization distributions more peaked
    and more compressible. PR #1508 actually implemented this as WARMDOWN_WD_MULT=2.0,
    which reduced entropy from 4.72 to 4.58 bits and increased zeros from 8.3% to 11.4%.

    This diagnostic measures the current weight distribution and estimates
    how much compression would improve if weights were more peaked.
    It does NOT modify the model — it's a diagnostic only.

    Returns a dict with per-tensor and aggregate statistics:
    - current entropy, zero-percentage
    - estimated compressed sizes at various simulated "peakedness" levels
    - theoretical savings from entropy regularization
    """
    results = {"tensors": {}, "aggregate": {}}

    total_symbols = 0
    total_entropy_bits = 0.0
    total_zeros = 0
    total_elements = 0

    # Per-tensor analysis
    for name, info in sorted(quant_meta.items()):
        if not isinstance(info, dict):
            continue
        qtype = info.get("type")
        if qtype not in ("int6", "int8"):
            continue

        q_key = name + ".q"
        if q_key not in quant_result:
            continue

        t = quant_result[q_key].numpy().flatten()
        n = len(t)
        total_elements += n

        # Current distribution
        if qtype == "int6":
            vals = t.astype(np.int16)
            zigzag = ((vals << 1) ^ (vals >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
        else:
            vals = t.astype(np.int16)
            zigzag = ((vals << 1) ^ (vals >> 15)).astype(np.int32)
            freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)

        probs = freqs / freqs.sum()
        probs_nz = probs[probs > 0]
        entropy = -np.sum(probs_nz * np.log2(probs_nz))

        n_zeros = int((t == 0).sum())
        pct_zeros = 100.0 * n_zeros / n

        total_symbols += n
        total_entropy_bits += entropy * n
        total_zeros += n_zeros

        results["tensors"][name] = {
            "type": qtype,
            "elements": n,
            "entropy_bps": round(entropy, 4),
            "zeros_pct": round(pct_zeros, 2),
            "theoretical_bytes": round(entropy * n / 8),
        }

    # Aggregate
    avg_entropy = total_entropy_bits / total_symbols if total_symbols > 0 else 0
    avg_zeros_pct = 100.0 * total_zeros / total_elements if total_elements > 0 else 0

    results["aggregate"] = {
        "total_symbols": total_symbols,
        "avg_entropy_bps": round(avg_entropy, 4),
        "avg_zeros_pct": round(avg_zeros_pct, 2),
        "theoretical_min_bytes": round(total_entropy_bits / 8),
    }

    # Simulate peakedness increase (as if entropy reg had been applied)
    # Model: multiply zero-bin probability by a factor, renormalize
    simulations = {}
    for zero_boost in [1.0, 1.25, 1.5, 2.0, 3.0]:
        sim_total_bits = 0.0
        for name, tinfo in results["tensors"].items():
            q_key = name + ".q"
            t = quant_result[q_key].numpy().flatten()
            if tinfo["type"] == "int6":
                vals = t.astype(np.int16)
                zigzag = ((vals << 1) ^ (vals >> 15)).astype(np.int32)
                freqs = np.bincount(zigzag, minlength=64)[:64].astype(np.float64)
            else:
                vals = t.astype(np.int16)
                zigzag = ((vals << 1) ^ (vals >> 15)).astype(np.int32)
                freqs = np.bincount(zigzag, minlength=256)[:256].astype(np.float64)
            # Boost zero bin (zigzag value 0 = original value 0)
            freqs[0] *= zero_boost
            probs = freqs / freqs.sum()
            probs_nz = probs[probs > 0]
            sim_entropy = -np.sum(probs_nz * np.log2(probs_nz))
            sim_total_bits += sim_entropy * len(t)

        sim_bytes = round(sim_total_bits / 8)
        savings_vs_current = round(total_entropy_bits / 8) - sim_bytes
        simulations[f"zero_boost_{zero_boost:.2f}x"] = {
            "theoretical_bytes": sim_bytes,
            "savings_vs_current": savings_vs_current,
        }

    results["entropy_reg_simulations"] = simulations
    return results


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

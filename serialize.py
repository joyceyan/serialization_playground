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
# EXPERIMENT: separate streams by dtype, transpose, zstd-22
# ==============================================================================

def _sorted_keys(d):
    return sorted(d.keys())


def encode_experiment(quant_result: dict[str, Tensor], quant_meta: dict[str, object]) -> bytes:
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
    N_CLUSTERS = 20
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

    # K-means clustering
    dist_matrix = np.array(int6_row_dists)
    centroids, labels = kmeans2(dist_matrix, N_CLUSTERS, minit="points", iter=20)

    # Build per-cluster frequency tables and encode
    cluster_compressed = []  # (cluster_id, compressed_bytes)
    cluster_freq_u16 = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if not mask.any():
            cluster_freq_u16.append(np.ones(64, dtype=np.uint16))
            cluster_compressed.append(b"")
            continue
        cluster_data = np.concatenate([int6_row_data[i] for i in range(len(labels)) if labels[i] == c])
        freqs = np.bincount(cluster_data, minlength=64)[:64].astype(np.float64)
        freqs = np.maximum(freqs, 1)
        freq_u16 = np.round(freqs / freqs.sum() * 65535).astype(np.uint16)
        freq_u16 = np.maximum(freq_u16, 1)
        probs = freq_u16.astype(np.float64) / freq_u16.astype(np.float64).sum()
        cluster_freq_u16.append(freq_u16)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(cluster_data, model)
        cluster_compressed.append(encoder.get_compressed().tobytes())

    # Compress labels (4-bit packed + LZMA)
    packed_labels = np.zeros((len(labels) + 1) // 2, dtype=np.uint8)
    for i in range(0, len(labels) - 1, 2):
        packed_labels[i // 2] = (labels[i] << 4) | labels[i + 1]
    if len(labels) % 2:
        packed_labels[-1] = labels[-1] << 4
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
    for c_data in cluster_compressed:
        ans_out += struct.pack("<I", len(c_data)) + c_data
    # Tensor row counts (to reconstruct row-to-tensor mapping)
    for nr in int6_tensor_nrows:
        ans_out += struct.pack("<H", nr)
    # Int8 tensors
    int8_freq_bytes = b"".join(f.tobytes() for _, f, _ in int8_parts)
    int8_freq_c = lzma.compress(int8_freq_bytes, format=lzma.FORMAT_RAW, filters=freq_filters) if int8_parts else b""
    ans_out += struct.pack("<H", len(int8_freq_c)) + int8_freq_c
    for c_data_bytes, _, numel in int8_parts:
        ans_out += struct.pack("<II", numel, len(c_data_bytes)) + c_data_bytes
    streams["int8"] = ans_out

    # fp16 stream: byte-shuffle (separate high and low bytes)
    fp16_parts = []
    for k in fp16_keys:
        t = quant_result[k].contiguous()
        fp16_parts.append(t.numpy().tobytes())
    if fp16_parts:
        raw = b"".join(fp16_parts)
        arr = np.frombuffer(raw, dtype=np.uint8)
        high = arr[0::2].tobytes()  # high bytes
        low = arr[1::2].tobytes()   # low bytes
        # Use raw LZMA2 with lp=1 (byte-shuffled data has 2-byte period)
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 1, "pb": 0}]
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

    # Encode metadata as JSON + LZMA — indexed format for compactness
    all_keys = _sorted_keys(quant_result)
    # Compact meta: derive meta_keys from tensor keys, encode types as string
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q") or k.endswith(".scale"):
            base = k.rsplit(".", 1)[0]
            if base not in meta_keys:
                meta_keys.append(base)
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

    # Abbreviate key names for compactness
    def _shorten(k):
        return k.replace("blocks.", "B").replace(".attn.", ".a.").replace(".mlp.", ".m.").replace(".weight", ".w").replace(".scale", ".s").replace(".proj.", ".p.")

    short_keys = [_shorten(k) for k in all_keys]

    header = json.dumps({
        "k": short_keys,
        "a": [512, 8, 4, 3, 11, 1024, 128],  # [D, H, KV, MLP, L, vocab, bigram_dim]
        "q": type_str,
    }, separators=(",", ":")).encode()
    # Raw LZMA2 for header too (saves ~32 bytes of .xz container overhead)
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


def _derive_shape(key, D, H, KV, MLP, L, vocab=1024, bigram_dim=128):
    """Derive tensor shape from key name and architecture params."""
    hd = D // H
    kv = KV * hd
    mlp = D * MLP
    bigram_vocab = vocab * 2

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

    if "q_gain" in key: return [H]
    if "attn_scale" in key: return [D]
    if "mlp_scale" in key: return [D]
    if "resid_mix" in key: return [2, D]
    if "skip_weight" in key: return [L // 2, D]
    if "bigram.proj" in key: return [D, bigram_dim]
    if "bigram.scale" in key: return []
    if "smear" in key and "gate" in key: return [D]
    return []


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
    # Decode indexed format
    def _expand(k):
        return k.replace("B", "blocks.").replace(".a.", ".attn.").replace(".m.", ".mlp.").replace(".w", ".weight").replace(".s", ".scale").replace(".p.", ".proj.")

    all_keys = [_expand(k) for k in header_json["k"]]
    # Derive shapes from architecture params
    arch = header_json["a"]
    D, H, KV, MLP, L = arch[:5]
    vocab = arch[5] if len(arch) > 5 else 1024
    bigram_dim = arch[6] if len(arch) > 6 else 128
    shapes_list = [_derive_shape(k, D, H, KV, MLP, L, vocab, bigram_dim) for k in all_keys]
    # Derive dtype classification from key suffixes (.q → int8, else → fp16)
    int8_indices = [i for i, k in enumerate(all_keys) if k.endswith(".q")]
    fp16_indices = [i for i, k in enumerate(all_keys) if not k.endswith(".q")]
    fp32_indices = []  # No fp32 in this format

    # Reconstruct quant_meta from type_str
    type_str = header_json.get("q", "")
    meta_keys = []
    for k in all_keys:
        if k.endswith(".q") or k.endswith(".scale"):
            base = k.rsplit(".", 1)[0]
            if base not in meta_keys:
                meta_keys.append(base)
        elif k not in meta_keys:
            meta_keys.append(k)
    type_map = {"p": "passthrough", "c": "passthrough_ctrl", "6": {"type": "int6"}, "8": {"type": "int8"}}
    quant_meta = {mk: type_map.get(type_str[i], "passthrough") for i, mk in enumerate(meta_keys)}

    header = {
        "int8_keys": [all_keys[i] for i in int8_indices],
        "fp16_keys": [all_keys[i] for i in fp16_indices],
        "fp32_keys": [all_keys[i] for i in fp32_indices],
        "shapes": {all_keys[i]: shapes_list[i] for i in range(len(all_keys))},
        "transposed": set(all_keys[i] for i in range(len(all_keys)) if len(shapes_list[i]) == 2),
        "byte_shuffle": True,
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
        # Unpack 4-bit labels
        labels_arr = np.frombuffer(labels_packed, dtype=np.uint8)
        labels = np.empty(n_rows, dtype=np.int32)
        for i in range(0, n_rows - 1, 2):
            labels[i] = labels_arr[i // 2] >> 4
            labels[i + 1] = labels_arr[i // 2] & 0x0F
        if n_rows % 2:
            labels[n_rows - 1] = labels_arr[n_rows // 2] >> 4

        # Parse cluster frequency tables
        cluster_probs = []
        for c in range(n_clusters):
            freq_u16 = np.frombuffer(all_freq_bytes[c * 128:(c + 1) * 128], dtype=np.uint16).astype(np.float64)
            probs = freq_u16 / freq_u16.sum()
            cluster_probs.append(probs)

        # Decode per-cluster ANS data
        cluster_data = {}
        for c in range(n_clusters):
            c_len = struct.unpack_from("<I", int8_block, boff)[0]; boff += 4
            if c_len == 0:
                cluster_data[c] = np.array([], dtype=np.int32)
                continue
            c_bytes = int8_block[boff:boff + c_len]; boff += c_len
            compressed = np.frombuffer(c_bytes, dtype=np.uint32)
            decoder = constriction.stream.stack.AnsCoder(compressed)
            model = constriction.stream.model.Categorical(cluster_probs[c], perfect=False)
            n_in_cluster = int(np.sum(labels == c))
            # Need total values in cluster — sum of row lengths
            # For now, decode all and split later
            # Actually we need the total number of values per cluster
            pass  # will compute below

        # Tensor row counts
        tensor_nrows = []
        for _ in range(n_int6):
            nr = struct.unpack_from("<H", int8_block, boff)[0]; boff += 2
            tensor_nrows.append(nr)

        # Compute per-row ncols from tensor shapes
        int6_keys_sorted = [k for k in sorted(quant_meta.keys())
                           if isinstance(quant_meta.get(k), dict) and quant_meta[k].get("type") == "int6"]
        # Map to actual tensor keys (.q suffix)
        int6_q_keys = [k for k in header["int8_keys"]
                       if any(k.startswith(mk) for mk in int6_keys_sorted)
                       and isinstance(quant_meta.get(k[:-2] if k.endswith(".q") else k), dict)
                       and quant_meta.get(k[:-2] if k.endswith(".q") else k, {}).get("type") == "int6"]

        row_ncols = []
        for ti, nr in enumerate(tensor_nrows):
            shape = header["shapes"][int6_q_keys[ti]]
            ncols = shape[-1] if len(shape) >= 2 else int(np.prod(shape))
            for _ in range(nr):
                row_ncols.append(ncols)

        # Now decode each cluster with the correct total symbol count
        cluster_decoded = {}
        # Re-read cluster data
        boff_clusters = 10 + 2 + freq_c_len + 2 + labels_c_len  # back to cluster data
        for c in range(n_clusters):
            c_len = struct.unpack_from("<I", int8_block, boff_clusters)[0]; boff_clusters += 4
            if c_len == 0:
                cluster_decoded[c] = np.array([], dtype=np.int32)
                continue
            c_bytes = int8_block[boff_clusters:boff_clusters + c_len]; boff_clusters += c_len
            compressed = np.frombuffer(c_bytes, dtype=np.uint32)
            decoder = constriction.stream.stack.AnsCoder(compressed)
            model = constriction.stream.model.Categorical(cluster_probs[c], perfect=False)
            total_syms = sum(row_ncols[i] for i in range(n_rows) if labels[i] == c)
            zigzag = decoder.decode(model, total_syms)
            cluster_decoded[c] = zigzag

        # Reconstruct rows in order
        cluster_offsets = {c: 0 for c in range(n_clusters)}
        int6_decoded = []
        for row_idx in range(n_rows):
            c = int(labels[row_idx])
            ncols = row_ncols[row_idx]
            off_c = cluster_offsets[c]
            row_zigzag = cluster_decoded[c][off_c:off_c + ncols]
            cluster_offsets[c] += ncols
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
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 0, "lp": 1, "pb": 0}]
        fp16_raw_shuffled = lzma.decompress(fp16_block, format=lzma.FORMAT_RAW, filters=filters)
    else:
        fp16_raw_shuffled = b""
    # fp32 block is optional (omitted when empty)
    if header["fp32_keys"] and off < len(blob):
        fp32_raw_shuffled = decomp.decompress(read_block())
    else:
        fp32_raw_shuffled = b""

    # Unshuffle fp16: high bytes then low bytes → interleave
    if fp16_raw_shuffled and header.get("byte_shuffle"):
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
    if fp32_raw_shuffled and header.get("byte_shuffle"):
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
    for label, keys in [("int8", header["int8_keys"]), ("fp16", header["fp16_keys"]), ("fp32", header["fp32_keys"])]:
        dt, raw, npdt = dtype_map[label]
        for k in keys:
            shape = header["shapes"][k]
            numel = 1
            for s in shape:
                numel *= s
            nbytes = numel * np.dtype(npdt).itemsize
            arr = np.frombuffer(raw, dtype=npdt, count=numel, offset=offsets[label])
            offsets[label] += nbytes
            if False:  # No transpose needed — ANS doesn't benefit from it
                pass
            else:
                t = torch.from_numpy(arr.copy()).reshape(shape)
            w[k] = t
    return w, header["meta"]


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

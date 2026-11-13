# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX full-vocabulary deterministic filter; full-shape RNG stays with caller.

Signed FP32 bit keys preserve the existing FlagGems radix order, including
signed zeros and NaN payloads. Native stable int32 sort uses the out overload;
FlagGems softmax/cumsum remain active. Unsupported metadata uses reference.
"""
import torch
import triton
import triton.language as tl

from flaggems_vllm.ops.top_k_top_p import apply_top_k_top_p as _reference_filter


@triton.jit
def _ordered_signed_key_kernel(Input, Output, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Input + offsets, mask=offsets < N, other=0)
    key = tl.where(value < 0, value ^ 0x7FFFFFFF, value)
    tl.store(Output + offsets, key, mask=offsets < N)


@triton.jit
def _final_p_mask_scatter(
    Out,
    Values,
    Indices,
    CDF,
    P,
    N: tl.constexpr,
    TOTAL,
    SPARSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < TOTAL
    row = offset // N
    column = offset % N
    cumulative = tl.load(CDF + offset, mask=valid, other=0.0)
    cutoff = 1.0 - tl.load(P + row, mask=valid, other=0.0)
    keep = ~(cumulative <= cutoff) | (column == N - 1)
    # Read logits as int32 to preserve all original FP32 payload bits.
    value = tl.load(Values + offset, mask=valid & keep, other=-8388608)
    if SPARSE:
        write = valid & keep & (value != -8388608)
        index = tl.load(Indices + offset, mask=write, other=0)
        tl.store(Out + row * N + index, value, mask=write)
    else:
        index = tl.load(Indices + offset, mask=valid, other=0)
        value = tl.where(keep, value, -8388608)
        tl.store(Out + row * N + index, value, mask=valid)


def supports_candidate(logits, k, p):
    return (
        logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.is_cuda
        and logits.is_contiguous()
        and 1 <= logits.shape[0] <= 64
        and logits.shape[1] == 248320
        and p is not None
        and p.dtype == torch.float32
        and p.device == logits.device
        and p.shape == (logits.shape[0],)
        and p.is_contiguous()
        and (
            k is None
            or (
                k.device == logits.device
                and k.ndim == 1
                and k.shape == (logits.shape[0],)
            )
        )
    )


def _filter_fp32(logits, k, p):
    bits = logits.view(torch.int32)
    keys = torch.empty_like(bits)
    _ordered_signed_key_kernel[(triton.cdiv(bits.numel(), 1024),)](
        bits, keys, bits.numel(), BLOCK=1024, num_warps=4
    )
    key_values = torch.empty_like(keys)
    indices = torch.empty(keys.shape, dtype=torch.int64, device=keys.device)
    torch.sort(keys, dim=-1, descending=False, stable=True, out=(key_values, indices))
    values = logits.gather(-1, indices)
    if k is not None:
        threshold = values.size(1) - k.to(torch.long)
        threshold = values.gather(1, threshold.unsqueeze(1))
        values.masked_fill_(values < threshold, -float("inf"))
    probabilities = values.softmax(dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1, out=probabilities)
    logits.fill_(-float("inf"))
    _final_p_mask_scatter[(triton.cdiv(logits.numel(), 1024),)](
        logits.view(torch.int32),
        values.view(torch.int32),
        indices,
        cumulative,
        p,
        logits.shape[1],
        logits.numel(),
        SPARSE=True,
        BLOCK=1024,
        num_warps=4,
    )
    return logits


def apply_top_k_top_p(logits, k, p):
    if supports_candidate(logits, k, p):
        return _filter_fp32(logits, k, p)
    return _reference_filter(logits, k, p)


def runtime_info():
    return {
        "candidate": "native_key_out_sparse1024",
        "provider": __name__,
        "helper": "_filter_fp32",
        "effective_config": {
            "key_block": 1024,
            "key_num_warps": 4,
            "scatter_block": 1024,
            "scatter_num_warps": 4,
            "sort_dtype": "int32",
            "sort_stable": True,
            "sort_overload": "aten::sort.values_stable",
            "sparse": True,
        },
        "scope": {
            "dtype": "float32",
            "ndim": 2,
            "contiguous": True,
            "vocab": 248320,
            "batch_min": 1,
            "batch_max": 64,
            "p_dtype": "float32",
            "p_shape": "B",
        },
        "fallback": _reference_filter.__module__ + "." + _reference_filter.__name__,
        "rng": "none; caller full[B,V] exponential path unchanged",
    }

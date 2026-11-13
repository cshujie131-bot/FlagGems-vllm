# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact variable-length FP32 top-k indices without TLE atomics.

Four byte-wise radix counts find the kth ordered floating-point key; a final
prefix scan collects larger keys and the required number of ties. This uses
the radix-select principle described by DeepSelect, with no copied CUDA/PTX.
Bounded tiles avoid register spills on MetaX C550.
Reference: deepseek-ai/DeepSelect bfa4507, docs/DeepSelect-deep-dive.md.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ordered_key(value):
    bits = tl.where(value == 0, 0, value.to(tl.uint32, bitcast=True))
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    return tl.where(value != value, 0xFFFFFFFF, key).to(tl.uint32)


@triton.jit
def _radix_topk_indices(
    logits,
    lengths,
    output,
    stride: tl.constexpr,
    width: tl.constexpr,
    k: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.minimum(tl.maximum(tl.load(lengths + row), 0), width)
    if length <= k:
        short_offsets = tl.arange(0, triton.next_power_of_2(k))
        tl.store(
            output + row * k + short_offsets,
            tl.where(short_offsets < length, short_offsets, -1),
            short_offsets < k,
        )
    else:
        lane = tl.arange(0, block)
        bins = tl.arange(0, 256)
        prefix = tl.full((), 0, tl.uint32)
        prefix_mask = tl.full((), 0, tl.uint32)
        remaining = tl.full((), k, tl.int32)
        # Refine only the boundary bin; larger bins contain final winners.
        for shift in tl.static_range(24, -1, -8):
            histogram = tl.zeros((256,), tl.int32)
            for tile in range(tl.cdiv(length, block)):
                offsets = tile * block + lane
                values = tl.load(
                    logits + row * stride + offsets,
                    offsets < length,
                    other=-float("inf"),
                )
                keys = _ordered_key(values)
                active = (offsets < length) & ((keys & prefix_mask) == prefix)
                digit = ((keys >> shift) & 255).to(tl.int32)
                histogram += tl.histogram(digit, 256, mask=active)
            suffix = tl.cumsum(histogram, 0, reverse=True)
            pivot = tl.max(tl.where(suffix >= remaining, bins, 0), 0)
            remaining -= tl.sum(tl.where(bins > pivot, histogram, 0), 0)
            prefix |= pivot.to(tl.uint32) << shift
            prefix_mask |= tl.full((), 255, tl.uint32) << shift

        # Lower original indices win ties. Each output location is written once.
        written = tl.full((), 0, tl.int32)
        for tile in range(tl.cdiv(length, block)):
            offsets = tile * block + lane
            values = tl.load(
                logits + row * stride + offsets, offsets < length, other=-float("inf")
            )
            keys = _ordered_key(values)
            equal = (offsets < length) & (keys == prefix)
            greater = (offsets < length) & (keys > prefix)
            tie_rank = tl.cumsum(equal.to(tl.int32), 0)
            take = greater | (equal & (tie_rank <= remaining))
            position = tl.cumsum(take.to(tl.int32), 0) - 1
            tl.store(output + row * k + written + position, offsets, take)
            remaining = tl.maximum(0, remaining - tl.sum(equal.to(tl.int32), 0))
            written += tl.sum(take.to(tl.int32), 0)


def radix_topk(logits, lengths, k=512, out=None):
    """Exact int32 top-k indices in ascending index order, with -1 padding.

    FP32 logits [rows,width<=32768], last stride 1; contiguous int32 lengths
    [rows] are clamped to [0,width] on device. k is in [1,4096]. Ties prefer
    lower indices, signed zeroes tie, and NaNs rank above numeric values.
    Caller-owned output must be contiguous and must not alias either input.
    No host reads of tensor contents or torch compute fallbacks are used.
    """
    if logits.ndim != 2:
        raise ValueError("radix_topk requires [rows,width] logits")
    if logits.dtype != torch.float32 or not logits.is_cuda:
        raise NotImplementedError("radix_topk supports accelerator FP32 logits")
    rows, width = logits.shape
    if width > 32768 or logits.stride(1) != 1:
        raise NotImplementedError("radix_topk requires width <=32768 and last stride 1")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 4096:
        raise ValueError("radix_topk supports integer k in [1,4096]")
    if (
        lengths.shape != (rows,)
        or lengths.dtype != torch.int32
        or lengths.device != logits.device
        or not lengths.is_contiguous()
    ):
        raise ValueError("radix_topk requires contiguous same-device int32 lengths")
    if out is None:
        out = torch.empty((rows, k), dtype=torch.int32, device=logits.device)
    elif (
        out.shape != (rows, k)
        or out.dtype != torch.int32
        or out.device != logits.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "radix_topk output must be contiguous same-device int32 [rows,k]"
        )
    if rows and (torch._C._overlaps(out, logits) or torch._C._overlaps(out, lengths)):
        raise ValueError("radix_topk output must not share input storage")
    if rows:
        with torch.cuda.device(logits.device):
            _radix_topk_indices[(rows,)](
                logits,
                lengths,
                out,
                logits.stride(0),
                width,
                k,
                min(2048, triton.next_power_of_2(max(width, 1))),
                num_warps=4,
                num_stages=1,
            )
    return out

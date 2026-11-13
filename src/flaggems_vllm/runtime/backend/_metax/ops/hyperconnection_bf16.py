# SPDX-License-Identifier: Apache-2.0
"""MetaX HC small operations preserving eager BF16 materialization boundaries.

Model projections, delayed-state scheduling and dispatch policy remain callers'
responsibility. This module does not pack weights or change residual ownership.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _grouped_norm(
    X, W, Y, H: tl.constexpr, G: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    x = tl.load(X + r * H + j, j < H, 0).to(tl.float32)
    w = tl.load(W + (r % G) * H + j, j < H, 0).to(tl.float32)
    variance = tl.sum(x * x, 0) / H
    normalized = x * tl.rsqrt(variance + EPS)
    y = normalized * (1.0 + w)
    tl.store(Y + r * H + j, y, j < H)


@tr.jit
def _gate_reduce(
    L,
    X,
    Y,
    H: tl.constexpr,
    G: tl.constexpr,
    SL: tl.constexpr,
    SX: tl.constexpr,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = linear // H
    j = linear % H
    valid = linear < N * H
    acc = tl.full((BLOCK,), 0, tl.float32)
    for g in tl.static_range(G):
        logits = tl.load(L + r * SL + g * H + j, valid, 0).to(tl.float32)
        x = tl.load(X + r * SX + g * H + j, valid, 0).to(tl.float32)
        gate = tl.sigmoid(logits).to(tl.bfloat16).to(tl.float32)
        product = (gate * x).to(tl.bfloat16).to(tl.float32)
        acc += product
    tl.store(Y + r * H + j, acc / G, valid)


@tr.jit
def _inject(
    L,
    B,
    R,
    Y,
    H: tl.constexpr,
    G: tl.constexpr,
    SL: tl.constexpr,
    SB: tl.constexpr,
    SR: tl.constexpr,
    BLOCK: tl.constexpr,
    N: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r = linear // (G * H)
    g = (linear // H) % G
    j = linear % H
    valid = linear < N * G * H
    logits = tl.load(L + r * SL + g, valid, 0).to(tl.float32)
    scaled = (logits / G).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(scaled).to(tl.bfloat16).to(tl.float32)
    weight = (2.0 * gate).to(tl.bfloat16).to(tl.float32)
    block = tl.load(B + r * SB + j, valid, 0).to(tl.float32)
    residual = tl.load(R + r * SR + g * H + j, valid, 0).to(tl.float32)
    product = (block * weight).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + (r * G + g) * H + j, residual + product, valid)


def _check(*xs):
    if not xs or any(
        x.device.type != "cuda" or x.dtype != torch.bfloat16 or not x.is_contiguous()
        for x in xs
    ):
        raise ValueError("HC requires contiguous BF16 accelerator tensors")
    if any(x.device != xs[0].device for x in xs):
        raise ValueError("HC tensors must share a device")


def grouped_norm(x, weight, groups, eps=1e-6, *, warps=4):
    _check(x, weight)
    if x.ndim != 2 or weight.ndim != 1 or weight.numel() != x.shape[1]:
        raise ValueError("Grouped norm shape mismatch")
    if groups <= 0 or x.shape[1] % groups or eps <= 0:
        raise ValueError("Invalid groups or epsilon")
    h = x.shape[1] // groups
    y = torch.empty_like(x)
    if x.numel():
        _grouped_norm[(x.shape[0] * groups,)](
            x,
            weight,
            y,
            h,
            groups,
            eps,
            tr.next_power_of_2(h),
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return y


def gate_reduce(logits, normed, groups, *, block=256, warps=4):
    _check(logits, normed)
    if (
        logits.ndim != 2
        or logits.shape != normed.shape
        or groups <= 0
        or logits.shape[1] % groups
    ):
        raise ValueError("Gate reduction shape mismatch")
    h = logits.shape[1] // groups
    y = torch.empty((logits.shape[0], h), dtype=logits.dtype, device=logits.device)
    if logits.numel():
        _gate_reduce[(tr.cdiv(logits.shape[0] * h, block),)](
            logits,
            normed,
            y,
            h,
            groups,
            logits.stride(0),
            normed.stride(0),
            block,
            logits.shape[0],
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return y


def inject(logits, block_output, residual, groups, *, block=256, warps=4):
    _check(logits, block_output, residual)
    if any(x.ndim != 2 for x in [logits, block_output, residual]) or groups <= 0:
        raise ValueError("Injection requires matrices and positive group count")
    n, h = block_output.shape
    if logits.shape != (n, groups) or residual.shape != (n, groups * h):
        raise ValueError("Injection shape mismatch")
    y = torch.empty_like(residual)
    if residual.numel():
        _inject[(tr.cdiv(n * groups * h, block),)](
            logits,
            block_output,
            residual,
            y,
            h,
            groups,
            logits.stride(0),
            block_output.stride(0),
            residual.stride(0),
            block,
            n,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return y


@tr.jit
def _divide_silu_bf16(X, Y, N, DIVISOR: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    # Keep the materialized BF16 division result before the existing SiLU math.
    divided = (x / DIVISOR).to(tl.bfloat16).to(tl.float32)
    value = tl.fdiv(divided, 1.0 + tl.exp(-divided))
    tl.store(Y + offsets, value, offsets < N)


def divide_silu_bf16(x, divisor, *, block=256, num_warps=4, out=None):
    """Fuse scalar division and SiLU while preserving BF16 intermediate rounding."""
    import math

    if x.dtype != torch.bfloat16 or x.device.type != "cuda" or not x.is_contiguous():
        raise ValueError("Expected contiguous accelerator BF16 input")
    if (
        not isinstance(divisor, (int, float))
        or not math.isfinite(divisor)
        or divisor <= 0
    ):
        raise ValueError("Divisor must be finite and positive")
    if out is None:
        out = torch.empty_like(x)
    if (
        out.shape != x.shape
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("Output must match contiguous input")
    if (
        out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        and out.data_ptr() != x.data_ptr()
    ):
        raise ValueError("Partial input/output overlap is not supported")
    if x.numel():
        _divide_silu_bf16[(tr.cdiv(x.numel(), block),)](
            x, out, x.numel(), float(divisor), BLOCK=block, num_warps=num_warps
        )
    return out

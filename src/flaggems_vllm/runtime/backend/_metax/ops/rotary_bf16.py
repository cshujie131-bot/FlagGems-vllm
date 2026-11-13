# SPDX-License-Identifier: Apache-2.0
"""BF16 rotary embedding with the eager multiply/add rounding boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_bf16(
    X,
    POS,
    CACHE,
    OUT,
    SX0: tl.constexpr,
    SX1: tl.constexpr,
    SX2: tl.constexpr,
    SP0: tl.constexpr,
    SP1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    SO2: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    CACHE_ROWS: tl.constexpr,
    AXES: tl.constexpr,
    NEOX: tl.constexpr,
    SEC0: tl.constexpr,
    SEC1: tl.constexpr,
    SEC2: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    FP32_COMPUTE: tl.constexpr,
    BLOCK: tl.constexpr,
    SLOTS=None,
    SSLOT: tl.constexpr = 0,
    WRITE_CACHE: tl.constexpr = False,
    OUTPUT_PAGE: tl.constexpr = 1,
    OUTPUT_BLOCK_STRIDE: tl.constexpr = 0,
    OUTPUT_TOKEN_STRIDE: tl.constexpr = 0,
    OUTPUT_CAPACITY: tl.constexpr = 0,
):
    flat = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = flat < T * H * D
    d = flat % D
    head = (flat // D) % H
    row = flat // (D * H)
    if WRITE_CACHE:
        slot = tl.load(SLOTS + row * SSLOT, valid, -1).to(tl.int64)
        valid = valid & (slot >= 0) & (slot < OUTPUT_CAPACITY)
        output_base = (
            (slot // OUTPUT_PAGE) * OUTPUT_BLOCK_STRIDE
            + (slot % OUTPUT_PAGE) * OUTPUT_TOKEN_STRIDE
            + head * SO1
        )
    else:
        output_base = row * SO0 + head * SO1
    half = R // 2
    frequency = d % half if NEOX else d // 2
    first = d < half if NEOX else d % 2 == 0
    a_dim = frequency if NEOX else 2 * frequency
    b_dim = frequency + half if NEOX else 2 * frequency + 1
    rotated = valid & (d < R)
    axis = tl.full((BLOCK,), 0, tl.int32)
    if AXES == 3:
        if INTERLEAVED:
            height = (frequency % 3 == 1) & (frequency < 3 * SEC1)
            width = (frequency % 3 == 2) & (frequency < 3 * SEC2)
        else:
            height = (frequency >= SEC0) & (frequency < SEC0 + SEC1)
            width = (frequency >= SEC0 + SEC1) & (frequency < SEC0 + SEC1 + SEC2)
        axis = tl.where(height, 1, tl.where(width, 2, 0))
    position = tl.load(POS + axis * SP0 + row * SP1, rotated, 0)
    position = tl.where(position < 0, position + CACHE_ROWS, position)
    position_valid = (position >= 0) & (position < CACHE_ROWS)
    cos = tl.load(
        CACHE + position.to(tl.int64) * SC0 + frequency * SC1,
        rotated & position_valid,
        0,
    ).to(tl.float32)
    sin = tl.load(
        CACHE + position.to(tl.int64) * SC0 + (frequency + half) * SC1,
        rotated & position_valid,
        0,
    ).to(tl.float32)
    a = tl.load(X + row * SX0 + head * SX1 + a_dim * SX2, rotated, 0).to(tl.float32)
    b = tl.load(X + row * SX0 + head * SX1 + b_dim * SX2, rotated, 0).to(tl.float32)
    lhs = tl.where(first, a, b) * cos
    rhs = tl.where(first, b, a) * sin
    if not FP32_COMPUTE:
        lhs = lhs.to(tl.bfloat16).to(tl.float32)
        rhs = rhs.to(tl.bfloat16).to(tl.float32)
    value = tl.where(first, lhs - rhs, lhs + rhs).to(tl.bfloat16)
    tail = tl.load(X + row * SX0 + head * SX1 + d * SX2, valid & (d >= R), 0)
    result = tl.where(d < R, value, tail)
    tl.store(OUT + output_base + d * SO2, result, valid)


def _geometry(x, positions, cos_sin_cache, rotary_dim, sections):
    if x.ndim != 3 or x.dtype != torch.bfloat16 or x.device.type != "cuda":
        raise ValueError("Expected accelerator BF16 [tokens, heads, dim]")
    t, h, d = x.shape
    if (
        h <= 0
        or d <= 0
        or d > 256
        or rotary_dim <= 0
        or rotary_dim > d
        or rotary_dim % 2
    ):
        raise ValueError("Unsupported rotary geometry")
    if (
        positions.dtype not in (torch.int32, torch.int64)
        or positions.device != x.device
    ):
        raise ValueError("Positions must be device integer tensors")
    if positions.shape == (t,):
        axes = 1
        sp0 = 0
        sp1 = positions.stride(0)
        sections = (rotary_dim // 2, 0, 0)
    elif positions.shape == (3, t):
        axes = 3
        sp0, sp1 = positions.stride()
    else:
        raise ValueError("Expected [tokens] or [3,tokens] positions")
    if (
        sections is None
        or len(sections) != 3
        or min(sections) < 0
        or sum(sections) != rotary_dim // 2
    ):
        raise ValueError("MRoPE sections must cover half the rotary dimension")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or not cos_sin_cache.shape[0]
        or cos_sin_cache.dtype != x.dtype
        or cos_sin_cache.device != x.device
    ):
        raise ValueError("Invalid cosine/sine cache")
    return t, h, d, axes, sp0, sp1, tuple(map(int, sections))


def _check_output_storage(out, x, cache):
    if not out.numel() or not x.numel():
        return
    if out.untyped_storage().data_ptr() in (
        x.untyped_storage().data_ptr(),
        cache.untyped_storage().data_ptr(),
    ):
        raise ValueError("Overlapping rotary output is not supported")


def rotary_embedding_bf16(
    x,
    positions,
    cos_sin_cache,
    rotary_dim,
    *,
    neox=True,
    sections=None,
    interleaved=False,
    fp32_compute=False,
    out=None,
):
    """Apply partial 1D/MRoPE; positions must index the supplied cache."""
    t, h, d, axes, sp0, sp1, sections = _geometry(
        x, positions, cos_sin_cache, rotary_dim, sections
    )
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("Output must match input")
    _check_output_storage(out, x, cos_sin_cache)
    if not t:
        return out
    _rotary_bf16[(triton.cdiv(t * h * d, 256),)](
        x,
        positions,
        cos_sin_cache,
        out,
        *x.stride(),
        sp0,
        sp1,
        *cos_sin_cache.stride(),
        *out.stride(),
        t,
        h,
        d,
        rotary_dim,
        cos_sin_cache.shape[0],
        axes,
        bool(neox),
        *sections,
        bool(interleaved),
        bool(fp32_compute),
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


def rotary_embedding_bf16_store(
    x,
    positions,
    cos_sin_cache,
    rotary_dim,
    slots,
    out_cache,
    *,
    neox=True,
    sections=None,
    interleaved=False,
    fp32_compute=False,
):
    """Rotate and write unique valid slots; negative slots leave the cache unchanged."""
    t, h, d, axes, sp0, sp1, sections = _geometry(
        x, positions, cos_sin_cache, rotary_dim, sections
    )
    if (
        slots.shape != (t,)
        or slots.dtype not in (torch.int32, torch.int64)
        or slots.device != x.device
    ):
        raise ValueError("Invalid rotary cache slots")
    if (
        out_cache.ndim != 4
        or out_cache.shape[2:] != (h, d)
        or not all(out_cache.shape)
        or out_cache.dtype != x.dtype
        or out_cache.device != x.device
    ):
        raise ValueError("Invalid rotary output cache")
    _check_output_storage(out_cache, x, cos_sin_cache)
    if not t:
        return out_cache
    _rotary_bf16[(triton.cdiv(t * h * d, 256),)](
        x,
        positions,
        cos_sin_cache,
        out_cache,
        *x.stride(),
        sp0,
        sp1,
        *cos_sin_cache.stride(),
        0,
        out_cache.stride(2),
        out_cache.stride(3),
        t,
        h,
        d,
        rotary_dim,
        cos_sin_cache.shape[0],
        axes,
        bool(neox),
        *sections,
        bool(interleaved),
        bool(fp32_compute),
        BLOCK=256,
        SLOTS=slots,
        SSLOT=slots.stride(0),
        WRITE_CACHE=True,
        OUTPUT_PAGE=out_cache.shape[1],
        OUTPUT_BLOCK_STRIDE=out_cache.stride(0),
        OUTPUT_TOKEN_STRIDE=out_cache.stride(1),
        OUTPUT_CAPACITY=out_cache.shape[0] * out_cache.shape[1],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out_cache

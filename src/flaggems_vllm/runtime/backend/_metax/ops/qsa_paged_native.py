# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX QSA virtual-page adapter over the installed MACA FlashAttention ABI."""

import torch
import triton as tr
import triton.language as tl
from flash_attn import flash_attn_varlen_func


@tr.jit
def scale_q(Q, OUTPUT, SQ0: tl.constexpr, SQ1: tl.constexpr, M: tl.constexpr):
    r = tl.program_id(0)
    h = tl.arange(0, 4)
    d = tl.arange(0, 256)
    q = tl.load(Q + r * SQ0 + h[:, None] * SQ1 + d[None, :], h[:, None] < 3, 0)
    q = (q * 0.0625 * 1.4426950408889634).to(q.dtype)
    tl.store(OUTPUT + (r * 3 + h[:, None]) * 256 + d[None, :], q, h[:, None] < 3)


@tr.jit
def apply_gate(
    Y,
    G,
    OUTPUT,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SG2: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    SO2: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.arange(0, 4)
    d = tl.arange(0, 256)
    y = tl.load(Y + (r * 3 + h[:, None]) * 256 + d[None, :], h[:, None] < 3, 0)
    g = tl.load(G + r * SG0 + h[:, None] * SG1 + d[None, :] * SG2, h[:, None] < 3, 0)
    out = (y * tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)).to(tl.bfloat16)
    tl.store(
        OUTPUT + r * SO0 + h[:, None] * SO1 + d[None, :] * SO2, out, h[:, None] < 3
    )


def native_qsa_attention(q, k, v, out, gate, workspace):
    if q.dtype != torch.bfloat16 or q.shape[1:] != (3, 256) or q.stride(-1) != 1:
        raise ValueError("Native QSA requires BF16 Q[rows,3,256]")
    if k.shape != v.shape or k.shape[2:] != (1, 256) or k.shape[1] % 4:
        raise ValueError("Native QSA requires four-token-compatible KV pages")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.stride(-1) != 1
        or v.stride(-1) != 1
        or k.stride(1) != 256
        or k.stride(0) != v.stride(0)
        or k.stride(0) % 1024
    ):
        raise ValueError("Unsupported native QSA KV strides")
    m = q.shape[0]
    step = k.stride(0) // 1024
    nblocks = (k.shape[0] - 1) * step + k.shape[1] // 4
    key = torch.as_strided(k, (nblocks, 4, 1, 256), (1024, 256, 256, 1))
    value = torch.as_strided(v, (nblocks, 4, 1, 256), (1024, 256, 256, 1))
    scale_q[(m,)](q, workspace["query"], *q.stride()[:2], m, num_warps=4)
    y = flash_attn_varlen_func(
        workspace["query"],
        key,
        value,
        workspace["cuq"],
        workspace["cu"],
        1,
        2051,
        dropout_p=0.0,
        softmax_scale=1 / 1.4426950408889634,
        causal=False,
        block_table=workspace["table"],
    )
    apply_gate[(m,)](y, gate, out, *gate.stride(), *out.stride(), num_warps=4)
    return out

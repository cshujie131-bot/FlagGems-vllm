# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX BF16 QSA score packing and invisible-tile skipping."""

import math

import torch
import triton as tr
import triton.language as tl


@tr.jit
def score_packed(
    Q,
    K,
    TAB,
    T2R,
    POS,
    LENS,
    ROWS,
    LOG,
    VIS,
    DIV,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    ST: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    NP: tl.constexpr,
    NR: tl.constexpr,
    PAGE: tl.constexpr,
    PG: tl.constexpr,
    BN: tl.constexpr,
    SKIP: tl.constexpr,
):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    qi = tl.arange(0, PG)
    rows = tl.load(ROWS + group * PG + qi)
    good = (rows >= 0) & (rows < M)
    safe = tl.maximum(rows, 0)
    reqs = tl.load(T2R + safe, good, 0)
    pos = tl.load(POS + safe, good, -1)
    lens = tl.load(LENS + reqs, good & (reqs >= 0) & (reqs < NR), 0)
    visible = tl.where(good, tl.minimum((pos + 1) // 4, lens // 4), 0)
    if tile == 0:
        tl.store(VIS + rows, visible, good)
    n = tile * BN + tl.arange(0, BN)
    maxvis = tl.max(visible, 0)
    if not SKIP or tile * BN < maxvis:
        req = tl.load(T2R + tl.load(ROWS + group * PG))
        page = n // PAGE
        off = n % PAGE
        phys = tl.load(TAB + req * ST + page, (n < C) & (n < maxvis) & (page < ST), -1)
        valid = (n < C) & (n < maxvis) & (phys >= 0) & (phys < NP)
        d = tl.arange(0, 128)
        rh = tl.arange(0, PG * 4)
        qr = tl.load(ROWS + group * PG + rh // 4)
        query = tl.load(
            Q + qr[:, None] * SQ0 + (rh[:, None] % 4) * SQ1 + d[None, :],
            (qr[:, None] >= 0) & (qr[:, None] < M),
            0,
        )
        keys = tl.load(
            K
            + tl.maximum(phys, 0)[None, :].to(tl.int64) * SK0
            + off[None, :] * SK1
            + d[:, None],
            valid[None, :],
            0.0,
        )
        dots = tl.dot(query, keys)
        scores = tl.sum(tl.reshape(tl.maximum(dots, 0.0), (PG, 4, BN)), 1) / DIV
        mask = good[:, None] & (n[None, :] < visible[:, None]) & valid[None, :]
        scores = tl.where(mask, scores, -float("inf"))
    else:
        scores = tl.full((PG, BN), -float("inf"), tl.float32)
    tl.store(
        LOG + rows[:, None] * C + n[None, :], scores, good[:, None] & (n[None, :] < C)
    )


@tr.jit
def score_decode(
    Q,
    K,
    TAB,
    MAP,
    POS,
    LENS,
    LOG,
    VIS,
    DIV,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    ST: tl.constexpr,
    M,
    C,
    NP,
    NR,
    PAGE: tl.constexpr,
    BN: tl.constexpr,
    DOT: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    n = tile * BN + tl.arange(0, BN)
    d = tl.arange(0, 128)
    req = tl.load(MAP + row)
    safe = tl.minimum(tl.maximum(req, 0), NR - 1)
    length = tl.load(LENS + safe, (req >= 0) & (req < NR), 0)
    pos = tl.load(POS + row)
    visible = tl.minimum((pos + 1) // 4, length // 4)
    if tile == 0:
        tl.store(VIS + row, visible)
    if tile * BN < visible:
        page = n // PAGE
        off = n % PAGE
        valid = (n < C) & (n < visible) & (req >= 0) & (req < NR) & (page < ST)
        phys = tl.load(TAB + safe * ST + page, valid, -1)
        valid &= (phys >= 0) & (phys < NP)
        keys = tl.load(
            K
            + tl.maximum(phys, 0)[None, :].to(tl.int64) * SK0
            + off[None, :] * SK1
            + d[:, None],
            valid[None, :],
            0.0,
        )
        if DOT:
            h = tl.arange(0, 16)
            query = tl.load(
                Q + row * SQ0 + h[:, None] * SQ1 + d[None, :], h[:, None] < 4, 0
            )
            value = tl.sum(tl.maximum(tl.dot(query, keys), 0.0), 0) / DIV
        else:
            kt = tl.trans(keys).to(tl.float32)
            value = tl.zeros((BN,), tl.float32)
            for h in tl.static_range(0, 4):
                query = tl.load(Q + row * SQ0 + h * SQ1 + d).to(tl.float32)
                value += tl.maximum(tl.sum(kt * query[None, :], 1), 0.0)
            value = value / DIV
        value = tl.where(valid, value, -float("inf"))
    else:
        value = tl.full((BN,), -float("inf"), tl.float32)
    tl.store(LOG + row * C + n, value, n < C)


def qsa_indexer_fast(
    q,
    k,
    table,
    mapping,
    positions,
    lengths,
    num_columns,
    pack_rows=None,
    *,
    decode_config=None,
    block_n=None,
    num_warps=None,
    num_stages=1,
):
    """BF16 QSA indexer; caller groups are same-request rows, prepared once per step."""
    if q.dtype != torch.bfloat16 or q.shape[1:] != (4, 128) or k.dtype != q.dtype:
        raise ValueError("MetaX fast QSA requires BF16 Q[rows,4,128]")
    m = q.shape[0]
    c = int(num_columns)
    logits = torch.empty((m, c), device=q.device, dtype=torch.float32)
    visible = torch.empty(m, device=q.device, dtype=torch.int32)
    if not m or not c:
        return logits, visible
    if pack_rows is not None:
        if pack_rows.ndim != 2 or pack_rows.shape[1] != 16:
            raise ValueError("QSA indexer groups must contain 16 query rows")
        bn = (128 if c <= 2048 else 64) if block_n is None else block_n
        score_packed[(pack_rows.shape[0], tr.cdiv(c, bn))](
            q,
            k,
            table,
            mapping,
            positions,
            lengths,
            pack_rows,
            logits,
            visible,
            math.sqrt(128),
            *q.stride()[:2],
            *k.stride()[:2],
            table.stride(0),
            m,
            c,
            k.shape[0],
            lengths.numel(),
            k.shape[1],
            16,
            bn,
            True,
            num_warps=8 if num_warps is None else num_warps,
            num_stages=num_stages,
        )
    else:
        bn, warps = (
            (64 if (c > 8192 and m >= 16) or (c > 2048 and m >= 32) else 32, 2)
            if m >= 8
            else (32, 4)
        )
        if decode_config is not None:
            bn, warps = decode_config
        if block_n is not None:
            bn = block_n
        if num_warps is not None:
            warps = num_warps
        score_decode[(m, tr.cdiv(c, bn))](
            q,
            k,
            table,
            mapping,
            positions,
            lengths,
            logits,
            visible,
            math.sqrt(128),
            *q.stride()[:2],
            *k.stride()[:2],
            table.stride(0),
            m,
            c,
            k.shape[0],
            lengths.numel(),
            k.shape[1],
            bn,
            m >= 8,
            num_warps=warps,
            num_stages=num_stages,
        )
    return logits, visible

# SPDX-License-Identifier: Apache-2.0
"""Exact top20 threshold and tail reconstruction; retain full-vocabulary softmax/cumsum.
Derived from the validated report37 candidate; caller supplies a host-verified k20 contract.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def sampling_top20_prepare(
    X, KV, KI, TV, TI, TH, NG, V: tl.constexpr, K: tl.constexpr = 20
):
    r = tl.program_id(0)
    j = tl.arange(0, 32)
    ok = j < K
    kk = tl.load(KV + r * K + j, ok, 2147483647)
    ii = tl.load(KI + r * K + j, ok, 2147483647)
    kth = tl.min(tl.where(ok, kk, 2147483647), 0)
    ix = tl.min(tl.where(ok & (kk == kth), ii, 2147483647), 0)
    th = tl.load(X + r * V + ix)
    fth = th.to(tl.float32, bitcast=True)
    vb = tl.load(X + r * V + ii, ok, 0)
    fv = vb.to(tl.float32, bitcast=True)
    greater = ok & (fv > fth)
    ng = tl.sum(greater.to(tl.int32), 0)
    dst = tl.sum(
        (
            (
                (kk[:, None] > kk[None, :])
                | ((kk[:, None] == kk[None, :]) & (ii[:, None] > ii[None, :]))
            )
            & greater[None, :]
        ).to(tl.int32),
        axis=1,
    )
    tl.store(TV + r * 32 + dst, vb, greater & (dst >= 0) & (dst < 32))
    tl.store(TI + r * 32 + dst, ii, greater & (dst >= 0) & (dst < 32))
    tl.store(TH + r, th)
    tl.store(NG + r, ng)


@tr.jit
def sampling_top20_count(X, TH, C, V: tl.constexpr, NC: tl.constexpr, T: tl.constexpr):
    r = tl.program_id(0)
    c = tl.program_id(1)
    j = c * T + tl.arange(0, T)
    valid = j < V
    bits = tl.load(X + r * V + j, valid, 0)
    x = bits.to(tl.float32, bitcast=True)
    th = tl.load(TH + r).to(tl.float32, bitcast=True)
    tie = valid & (x == th)
    a = tie & ((th != 0) | (bits < 0))
    b = tie & (th == 0) & (bits == 0)
    ab = bits & 0x7FFFFFFF
    special = valid & ((ab > 0x7F800000) | (bits == 0x7F800000))
    finite = valid & (ab < 0x7F800000)
    tl.store(C + (r * 4 + 0) * NC + c, tl.sum(a.to(tl.int32), 0))
    tl.store(C + (r * 4 + 1) * NC + c, tl.sum(b.to(tl.int32), 0))
    tl.store(C + (r * 4 + 2) * NC + c, tl.sum(special.to(tl.int32), 0))
    tl.store(C + (r * 4 + 3) * NC + c, tl.sum(finite.to(tl.int32), 0))


@tr.jit
def sampling_top20_prefix(C, P, M, NC: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, BC)
    valid = j < NC
    a = tl.load(C + (r * 4 + 0) * NC + j, valid, 0)
    b = tl.load(C + (r * 4 + 1) * NC + j, valid, 0)
    s = tl.load(C + (r * 4 + 2) * NC + j, valid, 0)
    f = tl.load(C + (r * 4 + 3) * NC + j, valid, 0)
    tl.store(P + (r * 2 + 0) * NC + j, tl.cumsum(a, 0) - a, valid)
    tl.store(P + (r * 2 + 1) * NC + j, tl.cumsum(b, 0) - b, valid)
    tl.store(M + r * 3 + 0, tl.sum(a, 0))
    tl.store(M + r * 3 + 1, tl.sum(b, 0))
    tl.store(M + r * 3 + 2, (tl.sum(s, 0) > 0) | (tl.sum(f, 0) == 0))


@tr.jit
def sampling_top20_make_tail(TV, TH, NG, M, S, V: tl.constexpr, T: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1) * T + tl.arange(0, T)
    valid = j < V
    ng = tl.load(NG + r)
    a = tl.load(M + r * 3)
    b = tl.load(M + r * 3 + 1)
    th = tl.load(TH + r)
    start = V - ng - a - b
    val = tl.full((T,), -8388608, tl.int32)
    tie = (j >= start) & (j < V - ng)
    zeroth = (th & 0x7FFFFFFF) == 0
    tieval = tl.where(zeroth, tl.where(j < start + a, -2147483648, 0), th)
    val = tl.where(tie, tieval, val)
    gt = j >= V - ng
    vb = tl.load(TV + r * 32 + j - (V - ng), valid & gt, 0)
    val = tl.where(gt, vb, val)
    tl.store(S + r * V + j, val, valid)


@tr.jit
def sampling_top20_write_ties(
    X, TH, NG, M, P, CDF, PP, V: tl.constexpr, NC: tl.constexpr, T: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    j = c * T + tl.arange(0, T)
    valid = j < V
    bits = tl.load(X + r * V + j, valid, 0)
    x = bits.to(tl.float32, bitcast=True)
    th = tl.load(TH + r).to(tl.float32, bitcast=True)
    ng = tl.load(NG + r)
    a = tl.load(M + r * 3)
    b = tl.load(M + r * 3 + 1)
    special = tl.load(M + r * 3 + 2) != 0
    tie = valid & (x == th)
    ta = tie & ((th != 0) | (bits < 0))
    tb = tie & (th == 0) & (bits == 0)
    pa = tl.cumsum(ta.to(tl.int32), 0) - 1 + tl.load(P + (r * 2 + 0) * NC + c)
    pb = tl.cumsum(tb.to(tl.int32), 0) - 1 + tl.load(P + (r * 2 + 1) * NC + c)
    slot = V - ng - a - b + tl.where(ta, pa, a + pb)
    cumulative = tl.load(CDF + r * V + slot, tie & ~special, 0.0)
    cut = 1.0 - tl.load(PP + r)
    keep = tie & ((cumulative > cut) | (slot == V - 1) | ((cumulative != cumulative)))
    out = tl.where(keep, bits, -8388608)
    out = tl.where(special, tl.where(x < th, -8388608, bits), out)
    tl.store(X + r * V + j, out, valid)


@tr.jit
def sampling_top20_write_greater(X, TV, TI, NG, M, CDF, PP, V: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, 32)
    ng = tl.load(NG + r)
    special = tl.load(M + r * 3 + 2) != 0
    valid = (j < ng) & ~special
    bits = tl.load(TV + r * 32 + j, valid, 0)
    ix = tl.load(TI + r * 32 + j, valid, 0)
    slot = V - ng + j
    c = tl.load(CDF + r * V + slot, valid, 0.0)
    keep = ~(c <= 1.0 - tl.load(PP + r)) | (slot == V - 1)
    tl.store(X + r * V + ix, bits, valid & keep)


@tr.jit
def sampling_top20_chunk_max(
    X,
    OV,
    OI,
    V: tl.constexpr,
    NC: tl.constexpr,
    T: tl.constexpr,
    KEEP: tl.constexpr = 20,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    j = c * T + tl.arange(0, T)
    valid = j < V
    b = tl.load(X + r * V + j, valid, 0)
    k = tl.where(b < 0, b ^ 0x7FFFFFFF, b)
    k = tl.where(valid, k, -2147483648)
    for z in range(KEEP):
        v = tl.max(k, 0)
        ix = tl.min(tl.where(valid & (k == v), j, 2147483647), 0)
        tl.store(OV + (r * NC + c) * KEEP + z, v)
        tl.store(OI + (r * NC + c) * KEEP + z, ix.to(tl.int64))
        valid = valid & (j != ix)
        k = tl.where(valid, k, -2147483648)


@tr.jit
def sampling_top20_merge_top(CV, CI, OV, OI, N: tl.constexpr, T: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, T)
    valid = j < N
    k = tl.load(CV + r * N + j, valid, -2147483648)
    ix = tl.load(CI + r * N + j, valid, 2147483647)
    for z in range(20):
        v = tl.max(tl.where(valid, k, -2147483648), 0)
        ii = tl.min(tl.where(valid & (k == v), ix, 2147483647), 0)
        tl.store(OV + r * 20 + z, v)
        tl.store(OI + r * 20 + z, ii)
        valid = valid & (ix != ii)


def select_top20(x):
    b, v = x.shape
    nc = tr.cdiv(v, 4096)
    ov = torch.empty((b, 20), dtype=torch.int32, device=x.device)
    oi = torch.empty((b, 20), dtype=torch.int64, device=x.device)
    cv = torch.empty((b, nc * 20), dtype=torch.int32, device=x.device)
    ci = torch.empty((b, nc * 20), dtype=torch.int64, device=x.device)
    sampling_top20_chunk_max[(b, nc)](
        x.view(torch.int32), cv, ci, v, nc, 4096, 20, num_warps=4
    )
    sampling_top20_merge_top[(b,)](
        cv, ci, ov, oi, nc * 20, tr.next_power_of_2(nc * 20), num_warps=8
    )
    return ov, oi


def apply_top20_top_p(x, p):
    tile = 4096
    warps = 4
    b, v = x.shape
    ov, oi = select_top20(x)
    tv = torch.empty((b, 32), dtype=torch.int32, device=x.device)
    ti = torch.empty((b, 32), dtype=torch.int64, device=x.device)
    th = torch.empty((b,), dtype=torch.int32, device=x.device)
    ng = torch.empty_like(th)
    sampling_top20_prepare[(b,)](
        x.view(torch.int32), ov, oi, tv, ti, th, ng, v, num_warps=4
    )
    nc = tr.cdiv(v, tile)
    cnt = torch.empty((b, 4, nc), dtype=torch.int32, device=x.device)
    pref = torch.empty((b, 2, nc), dtype=torch.int32, device=x.device)
    meta = torch.empty((b, 3), dtype=torch.int32, device=x.device)
    sampling_top20_count[(b, nc)](
        x.view(torch.int32), th, cnt, v, nc, tile, num_warps=warps
    )
    sampling_top20_prefix[(b,)](
        cnt, pref, meta, nc, tr.next_power_of_2(nc), num_warps=4
    )
    sorted_values = torch.empty_like(x)
    sampling_top20_make_tail[(b, nc)](
        tv, th, ng, meta, sorted_values.view(torch.int32), v, tile, num_warps=warps
    )
    probs = sorted_values.softmax(-1)
    torch.cumsum(probs, -1, out=probs)
    sampling_top20_write_ties[(b, nc)](
        x.view(torch.int32), th, ng, meta, pref, probs, p, v, nc, tile, num_warps=warps
    )
    sampling_top20_write_greater[(b,)](
        x.view(torch.int32), tv, ti, ng, meta, probs, p, v, num_warps=4
    )
    return x

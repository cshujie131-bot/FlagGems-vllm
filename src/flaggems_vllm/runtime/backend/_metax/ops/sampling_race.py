# SPDX-License-Identifier: Apache-2.0
"""Fuse the existing FP32 probability/noise division and argmax.
Probability normalization and the full-vocabulary RNG are kept unchanged.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def sampling_race_partial(
    P, Q, OV, OI, V: tl.constexpr, NC: tl.constexpr, T: tl.constexpr
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    j = c * T + tl.arange(0, T)
    valid = j < V
    p = tl.load(P + r * V + j, valid, 0.0)
    q = tl.load(Q + r * V + j, valid, 1.0)
    v = p / q
    nan = valid & (v != v)
    hasnan = tl.sum(nan.to(tl.int32), 0) > 0
    mx = tl.max(tl.where(valid & ~nan, v, -float("inf")), 0)
    ix = tl.min(tl.where(tl.where(hasnan, nan, valid & (v == mx)), j, 2147483647), 0)
    tl.store(OV + r * NC + c, tl.where(hasnan, float("nan"), mx))
    tl.store(OI + r * NC + c, ix)


@tr.jit
def sampling_race_finish(OV, OI, OUTPUT, NC: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, BC)
    valid = j < NC
    v = tl.load(OV + r * NC + j, valid, -float("inf"))
    ix = tl.load(OI + r * NC + j, valid, 2147483647)
    nan = valid & (v != v)
    hasnan = tl.sum(nan.to(tl.int32), 0) > 0
    mx = tl.max(tl.where(valid & ~nan, v, -float("inf")), 0)
    ans = tl.min(tl.where(tl.where(hasnan, nan, valid & (v == mx)), ix, 2147483647), 0)
    tl.store(OUTPUT + r, ans)


def sample(probs, noise, tile=4096, warps=4):
    b, v = probs.shape
    nc = tr.cdiv(v, tile)
    vals = torch.empty((b, nc), dtype=torch.float32, device=probs.device)
    ids = torch.empty((b, nc), dtype=torch.int32, device=probs.device)
    out = torch.empty((b,), dtype=torch.int64, device=probs.device)
    sampling_race_partial[(b, nc)](
        probs, noise, vals, ids, v, nc, tile, num_warps=warps
    )
    sampling_race_finish[(b,)](vals, ids, out, nc, tr.next_power_of_2(nc), num_warps=4)
    return out

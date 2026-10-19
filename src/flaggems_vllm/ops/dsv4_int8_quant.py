"""C550 dynamic symmetric BF16 W8A8 specialization based on scaled_int8_quant.

Matches the deployed native arithmetic order. The scale's decimal factor is a
compatibility expression inferred from measurements; exhaustive BF16 scale tests
are a required acceptance gate, not an assumption about ideal 1/127 arithmetic.
"""

import torch, triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language import core

# Extend only a private extern-symbol entry; preserve all existing mappings.
assert (
    core.nv_to_maca_map.get("__dsv4_mxc_rcp", "llvm.mxc.rcp.f32") == "llvm.mxc.rcp.f32"
)
core.nv_to_maca_map["__dsv4_mxc_rcp"] = "llvm.mxc.rcp.f32"


@core.extern
def mxc_rcp(x, _builder=None):
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("__dsv4_mxc_rcp", tl.float32)},
        is_pure=True,
        _builder=_builder,
    )


# FlagTree 3.1 dependency visitor recognizes externs by the Triton namespace.
# Add a new intrinsic wrapper; no existing function or compiler pass is replaced.
mxc_rcp.__module__ = "triton.language.extra.libdevice"
libdevice.dsv4_mxc_rcp = mxc_rcp
from flaggems_vllm.ops.dsv4_launch import launch


@triton.jit
def dynamic_int8_native_order(
    X, Y, S, M: tl.constexpr, K: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    x = tl.load(
        X + rows[:, None] * K + ks[None, :], (rows[:, None] < M) & (ks[None, :] < K), 0
    ).to(tl.float32)
    a = tl.max(tl.abs(x), 1)
    if K <= 8192:
        # SDK __fdividef uses the hardware reciprocal, not libdevice rcp_rn.
        denormal = a < 1.1754943508222875e-38
        den = tl.where(denormal, a * 16777216.0, a)
        numerator = tl.where(denormal, 127.0 * 16777216.0, 127.0)
        inv = tl.where(a > 0, mxc_rcp(den) * numerator, 0.0)
        z = x * inv[:, None]
        s = (a.to(tl.float64) * tl.full((), 0.0078740157, tl.float64)).to(tl.float32)
    else:
        inv = tl.where(a > 0, 127.0 / a, 0.0)
        z = x * inv[:, None]
        s = a / 127.0
    q = libdevice.rint(z)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8)
    tl.store(
        Y + rows[:, None] * K + ks[None, :], q, (rows[:, None] < M) & (ks[None, :] < K)
    )
    tl.store(S + rows, s, rows < M)


def quant(x, bm=None, nw=None):
    m, k = x.shape
    if bm is None:
        bm = 8 if k <= 512 and m >= 128 else (4 if k <= 1024 and m >= 128 else 1)
    if nw is None:
        if k > 8192:
            nw = 8
        elif k == 4096 and m <= 64:
            nw = 16 if m == 1 else 8
        else:
            nw = 4
    q = torch.empty_like(x, dtype=torch.int8)
    s = torch.empty((m, 1), device=x.device, dtype=torch.float32)
    if m:
        launch(
            dynamic_int8_native_order,
            (triton.cdiv(m, bm),),
            (x, q, s, m, k, triton.next_power_of_2(k), bm),
            num_warps=nw,
            num_stages=1,
        )
    return q, s, None

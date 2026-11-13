# SPDX-License-Identifier: Apache-2.0
"""QSA component candidates. Numerical structure references Apache-2.0 FGV QSA.
Mode fold4: local 3->4 heads, duplicate Q into four V-quarter rows.
Mode simt4: four head rows, scalar reductions without MMA M16.
Mode pack4: four queries sharing request and selected-column layout, 3->4 heads each.
Caller-owned graph workspaces; optional single-split direct output on MetaX.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language import core


@core.extern
def _metax_rcp_fp32(x, _semantic=None):
    # Match the reciprocal emitted by the original C550 reduction kernel.
    # A correctly-rounded division/reciprocal changes BF16 tie cases here.
    return core.extern_elementwise(
        "",
        "",
        [x],
        {
            (core.dtype("fp32"),): ("llvm.mxc.rcp.f32", core.dtype("fp32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@tr.jit
def qsa_packed_partial(
    Q,
    K,
    V,
    IDX,
    TAB,
    MAP,
    ROWS,
    PO,
    PM,
    PL,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SI0: tl.constexpr,
    ST0: tl.constexpr,
    NP: tl.constexpr,
    B: tl.constexpr,
    WIDTH: tl.constexpr,
    S: tl.constexpr,
    MODE: tl.constexpr,
    BN: tl.constexpr,
    SKIP: tl.constexpr = False,
    WARP_DUMMY: tl.constexpr = 0,
    D: tl.constexpr = 256,
    PAGE: tl.constexpr = 16,
    G=None,
    O=None,
    SG0: tl.constexpr = 0,
    SG1: tl.constexpr = 0,
    SG2: tl.constexpr = 0,
    SO0: tl.constexpr = 0,
    SO1: tl.constexpr = 0,
    SO2: tl.constexpr = 0,
    DIRECT_OUT: tl.constexpr = False,
    GATE: tl.constexpr = False,
    PREFILL_OPT: tl.constexpr = False,
):
    tl.static_assert(not PREFILL_OPT or (MODE == 1 and SKIP))
    group = tl.program_id(0)
    split = tl.program_id(1)
    if MODE == 0:  # simt4
        HH: tl.constexpr = 4
        VD: tl.constexpr = 256
    else:
        HH: tl.constexpr = 16
        if MODE == 1:
            VD: tl.constexpr = 64
        else:
            VD: tl.constexpr = 256
    rh = tl.arange(0, HH)
    if MODE == 2:
        qr = tl.load(ROWS + group * 4 + rh // 4)
        h = rh % 4
        req = tl.load(MAP + tl.load(ROWS + group * 4))
    else:
        qr = tl.full((HH,), group, tl.int32)
        h = rh % 4 if MODE == 1 else rh
        req = tl.load(MAP + group)
    valid_h = (qr >= 0) & (qr < B) & (h < 3)
    if PREFILL_OPT:
        bound_cols = tl.arange(0, tr.next_power_of_2(WIDTH))
        bound_ids = tl.load(IDX + group * SI0 + bound_cols, bound_cols < WIDTH, -1)
        logical_end = tl.max(tl.where(bound_ids >= 0, bound_cols + 1, 0), 0)
    dd = tl.arange(0, D)
    q = tl.load(
        Q + qr[:, None] * SQ0 + h[:, None] * SQ1 + dd[None, :], valid_h[:, None], 0
    )
    q = (q * (D**-0.5) * 1.4426950408889634).to(q.dtype)
    m = tl.full((HH,), -1.0e20, tl.float32)
    denominator = tl.zeros((HH,), tl.float32)
    acc = tl.zeros((HH, VD), tl.float32)
    nn = tl.arange(0, BN)
    PER: tl.constexpr = tr.cdiv(WIDTH, S)
    if PREFILL_OPT:
        local_end = tl.minimum(PER, tl.maximum(logical_end - split * PER, 0))
        tile_count = (local_end + BN - 1) // BN
    else:
        tile_count = tr.cdiv(PER, BN)
    for start in range(tile_count):
        cols = split * PER + start * BN + nn
        vc = (cols < WIDTH) & (start * BN + nn < PER)
        if MODE == 2:
            raw = tl.load(
                IDX + qr[:, None] * SI0 + cols[None, :],
                valid_h[:, None] & vc[None, :],
                -1,
            )
            logical = tl.max(raw, 0)
            vm = valid_h[:, None] & (raw >= 0) & (raw == logical[None, :]) & vc[None, :]
        else:
            logical = tl.load(IDX + group * SI0 + cols, vc, -1)
            vm = valid_h[:, None] & (logical[None, :] >= 0) & vc[None, :]
        page = tl.maximum(logical, 0) // PAGE
        off = tl.maximum(logical, 0) % PAGE
        physical = tl.load(
            TAB + req * ST0 + page, (logical >= 0) & vc & (page < ST0), -1
        )
        ok = (logical >= 0) & vc & (physical >= 0) & (physical < NP)
        vm = vm & ok[None, :]
        if not SKIP or tl.sum(ok.to(tl.int32), 0) > 0:
            kk = tl.load(
                K
                + physical[None, :].to(tl.int64) * SK0
                + off[None, :] * SK1
                + dd[:, None],
                ok[None, :],
                0,
            )
            if MODE == 0:
                scores = tl.sum(
                    q.to(tl.float32)[:, None, :]
                    * tl.trans(kk).to(tl.float32)[None, :, :],
                    2,
                )
            else:
                scores = tl.dot(q, kk)
            scores = tl.where(vm, scores, -1.0e20)
            nm = tl.maximum(m, tl.max(scores, 1))
            alpha = tl.exp2(m - nm)
            pp = tl.where(vm, tl.exp2(scores - nm[:, None]), 0.0)
            if MODE == 1:
                joint = tl.arange(0, 4 * BN)
                tok = joint // 4
                part = joint % 4
                if PREFILL_OPT:
                    pj = tl.reshape(
                        tl.where(
                            (rh[:, None, None] // 4) == tl.arange(0, 4)[None, None, :],
                            pp[:, :, None],
                            0.0,
                        ),
                        (16, 4 * BN),
                    ).to(q.dtype)
                    jp = tl.reshape(
                        tl.broadcast_to(physical[:, None], (BN, 4)), (4 * BN,)
                    )
                    jo = tl.reshape(tl.broadcast_to(off[:, None], (BN, 4)), (4 * BN,))
                    jok = tl.reshape(tl.broadcast_to(ok[:, None], (BN, 4)), (4 * BN,))
                else:
                    p0 = tl.where((rh[:, None] // 4) == 0, pp, 0.0)
                    p1 = tl.where((rh[:, None] // 4) == 1, pp, 0.0)
                    p2 = tl.where((rh[:, None] // 4) == 2, pp, 0.0)
                    p3 = tl.where((rh[:, None] // 4) == 3, pp, 0.0)
                    pj = tl.reshape(
                        tl.join(tl.join(p0, p2), tl.join(p1, p3)), (16, 4 * BN)
                    ).to(q.dtype)
                    jc = split * PER + start * BN + tok
                    jv = (jc < WIDTH) & (start * BN + tok < PER)
                    jl = tl.load(IDX + group * SI0 + jc, jv, -1)
                    jpage = tl.maximum(jl, 0) // PAGE
                    jo = tl.maximum(jl, 0) % PAGE
                    jp = tl.load(
                        TAB + req * ST0 + jpage, (jl >= 0) & jv & (jpage < ST0), -1
                    )
                    jok = (jl >= 0) & jv & (jp >= 0) & (jp < NP)
                dv = tl.arange(0, 64)
                vv = tl.load(
                    V
                    + jp[:, None].to(tl.int64) * SV0
                    + jo[:, None] * SV1
                    + part[:, None] * 64
                    + dv[None, :],
                    jok[:, None],
                    0,
                )
                acc = tl.dot(pj, vv, acc=acc * alpha[:, None])
            else:
                vv = tl.load(
                    V
                    + physical[:, None].to(tl.int64) * SV0
                    + off[:, None] * SV1
                    + dd[None, :],
                    ok[:, None],
                    0,
                )
                if MODE == 0:
                    acc = acc * alpha[:, None] + tl.sum(
                        pp.to(q.dtype).to(tl.float32)[:, :, None]
                        * vv.to(tl.float32)[None, :, :],
                        1,
                    )
                else:
                    acc = tl.dot(pp.to(q.dtype), vv, acc=acc * alpha[:, None])
            denominator = denominator * alpha + tl.sum(pp, 1)
            m = nm
    if MODE == 1:
        dv = tl.arange(0, 64)
        destd = (rh[:, None] // 4) * 64 + dv[None, :]
        stats = valid_h & (rh < 4)
    else:
        dv = tl.arange(0, 256)
        destd = dv[None, :]
        stats = valid_h
    if DIRECT_OUT:
        # Match single-split reduce's division and both BF16 gate boundaries.
        inv_l = _metax_rcp_fp32(denominator)
        out = tl.where(denominator[:, None] > 0, acc * inv_l[:, None], 0.0)
        if GATE:
            gate = tl.load(
                G + qr[:, None] * SG0 + h[:, None] * SG1 + destd * SG2,
                valid_h[:, None],
                0,
            )
            out = (
                out.to(tl.bfloat16) * tl.sigmoid(gate.to(tl.float32)).to(tl.bfloat16)
            ).to(tl.bfloat16)
        tl.store(
            O + qr[:, None] * SO0 + h[:, None] * SO1 + destd * SO2,
            out,
            valid_h[:, None],
        )
    else:
        offsets = ((qr[:, None] * 3 + h[:, None]) * S + split) * D + destd
        tl.store(PO + offsets, acc, valid_h[:, None])
        tl.store(PM + (qr * 3 + h) * S + split, m, stats)
        tl.store(PL + (qr * 3 + h) * S + split, denominator, stats)


@tr.jit
def qsa_packed_reduce(
    PO,
    PM,
    PL,
    G,
    O,
    S: tl.constexpr,
    BS: tl.constexpr,
    SG0: tl.constexpr,
    SG1: tl.constexpr,
    SG2: tl.constexpr,
    SO0: tl.constexpr,
    SO1: tl.constexpr,
    SO2: tl.constexpr,
    D: tl.constexpr = 256,
    GATE: tl.constexpr = True,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, D)
    m = tl.load(PM + (row * 3 + h) * S + s, s < S, -1.0e20)
    denominator = tl.load(PL + (row * 3 + h) * S + s, s < S, 0)
    mm = tl.max(m, 0)
    scale = tl.exp2(m - mm)
    den = tl.sum(denominator * scale, 0)
    po = tl.load(
        PO + ((row * 3 + h) * S + s[:, None]) * D + d[None, :], s[:, None] < S, 0
    )
    num = tl.sum(po * scale[:, None], 0)
    out = tl.where(den > 0, num * _metax_rcp_fp32(den), 0.0)
    if GATE:
        g = tl.load(G + row * SG0 + h * SG1 + d * SG2)
        out = (out.to(tl.bfloat16) * tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16)).to(
            tl.bfloat16
        )
    tl.store(O + row * SO0 + h * SO1 + d * SO2, out)


class Candidate:
    def __init__(
        self,
        q,
        mode,
        splits,
        bn,
        warps,
        stages,
        rows=None,
        reduce_warps=4,
        direct_output=None,
        collect_metadata=True,
        prefill_optimization=False,
    ):
        self.skip = mode.endswith("_skip")
        mode = mode.removesuffix("_skip")
        self.mode = {"simt4": 0, "fold4": 1, "pack4": 2, "standard16": 3}[mode]
        self.s = splits
        self.bn = bn
        self.w = warps
        self.st = stages
        self.rw = reduce_warps
        self.rows = rows
        self.prefill_optimization = bool(prefill_optimization)
        if self.prefill_optimization and (
            self.mode != 1
            or not self.skip
            or q.device.type != "cuda"
            or q.dtype != torch.bfloat16
            or q.ndim != 3
            or q.shape[1:] != (3, 256)
        ):
            raise ValueError(
                "Prefill optimization requires CUDA BF16 fold4_skip Q[rows, 3, 256]"
            )
        supported = splits == 1 and self.mode == 1
        self.direct_output = supported if direct_output is None else bool(direct_output)
        if self.direct_output and not supported:
            raise ValueError("Direct output requires fold4 with one split")
        self.parts = self.max = self.sum = None
        if not self.direct_output:
            self.parts = torch.empty(
                (*q.shape[:2], splits, 256), dtype=torch.float32, device=q.device
            )
            self.max = torch.empty(
                (*q.shape[:2], splits), dtype=torch.float32, device=q.device
            )
            self.sum = torch.empty_like(self.max)
        self.collect_metadata = bool(collect_metadata)
        self.meta = []

    def __call__(self, q, k, v, idx, table, mapping, out, gate):
        b = q.shape[0]
        rows = self.rows if self.rows is not None else mapping
        groups = rows.shape[0] if self.mode == 2 else b
        z = qsa_packed_partial[(groups, self.s)](
            q,
            k,
            v,
            idx,
            table,
            mapping,
            rows,
            self.parts,
            self.max,
            self.sum,
            *q.stride()[:2],
            *k.stride()[:3],
            *v.stride()[:3],
            idx.stride(0),
            table.stride(0),
            k.shape[0],
            b,
            idx.shape[1],
            self.s,
            self.mode,
            self.bn,
            SKIP=self.skip,
            PAGE=k.shape[1],
            G=gate,
            O=out,
            SG0=gate.stride(0) if gate is not None else 0,
            SG1=gate.stride(1) if gate is not None else 0,
            SG2=gate.stride(2) if gate is not None else 0,
            SO0=out.stride(0),
            SO1=out.stride(1),
            SO2=out.stride(2),
            DIRECT_OUT=self.direct_output,
            GATE=gate is not None,
            PREFILL_OPT=self.prefill_optimization,
            num_warps=self.w,
            num_stages=self.st,
        )
        launches = [z] if self.collect_metadata else None
        if not self.direct_output:
            y = qsa_packed_reduce[(b, 3)](
                self.parts,
                self.max,
                self.sum,
                gate,
                out,
                self.s,
                tr.next_power_of_2(self.s),
                *(gate.stride() if gate is not None else (0, 0, 0)),
                *out.stride(),
                GATE=gate is not None,
                num_warps=self.rw,
                num_stages=1,
            )
            if launches is not None:
                launches.append(y)
        if (
            self.collect_metadata
            and not self.meta
            and not torch.cuda.is_current_stream_capturing()
        ):
            self.meta = [
                {
                    "n_regs": getattr(x, "n_regs", None),
                    "n_spills": getattr(x, "n_spills", None),
                    "shared": getattr(x.metadata, "shared", None),
                }
                for x in launches
            ]
        return out

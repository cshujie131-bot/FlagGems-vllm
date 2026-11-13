# SPDX-License-Identifier: Apache-2.0
"""BF16 MCTlass C ABI adapter with immutable unit scales and reusable descriptors.

Numerics stay in the installed vendor library. Inputs/outputs are borrowed per
call, not retained by the shape cache. The caller owns tensor/stream lifetimes.
"""
import ctypes as C
import math

import torch

P, I, U64, I64, SZ = C.c_void_p, C.c_int, C.c_uint64, C.c_int64, C.c_size_t
_LIB = C.CDLL("/opt/maca/lib/libmctlassEx.so")
_SIGNATURES = {
    "HandleCreate": [C.POINTER(P)],
    "HandleDestroy": [P],
    "CreateDesc": [C.POINTER(P)],
    "DestroyDesc": [P],
    "DescSetAttribute": [P, I, P, SZ],
    "MatrixLayoutCreate": [C.POINTER(P), I, U64, U64, I64],
    "MatrixLayoutDestroy": [P],
    "MatrixLayoutSetAttribute": [P, I, P, SZ],
    "FuseMoeDescCreate": [C.POINTER(P), P, P, P, P, I, I, C.c_bool],
    "FuseMoeDescDestroy": [P],
    "FuseMoeDescSetAttribute": [P, I, P, SZ],
    "FuseMoeGemmGetBlocksizeM": [P, P, P, P, P, I, C.POINTER(I), C.POINTER(I)],
    "FuseMoeGemmBasic": [P, P, P, P, P, P, P, P, P, C.POINTER(I), P, SZ, P],
    "UniversalGemmBasic": [P, P, P, P, P, P, P, P, C.POINTER(I), P, SZ, P],
}
for _name, _args in _SIGNATURES.items():
    _fn = getattr(_LIB, "mctlassEx" + _name)
    _fn.argtypes, _fn.restype = _args, I

_PLANS, _ONES, _IDENTITIES = {}, {}, {}
_STATS = {
    "prepare": 0,
    "cache_hit": 0,
    "capture_miss": 0,
    "moe": 0,
    "linear": 0,
    "errors": 0,
}


def _call(name, *args):
    code = getattr(_LIB, "mctlassEx" + name)(*args)
    if code:
        _STATS["errors"] += 1
        raise RuntimeError(f"mctlassEx{name}: status {code}")
    return code


def _ptr(tensor):
    return P(tensor.data_ptr()) if tensor is not None else P()


def _attr(name, obj, index, value):
    _call(name, obj, index, C.byref(value), C.sizeof(value))


def _ones(count, device):
    key = (device.index, count)
    if key not in _ONES:
        _ONES[key] = torch.ones(count, device=device, dtype=torch.float32)
    return _ONES[key]


def supported(a, b, c, topk=0):
    return (
        a.ndim == 2
        and b.ndim == (3 if topk else 2)
        and a.device.type == "cuda"
        and a.device == b.device == c.device
        and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.shape[0] > 0
        and a.shape[1] == b.shape[-1]
        and c.numel() == a.shape[0] * max(topk, 1) * b.shape[-2]
        and a.is_contiguous()
        and b.is_contiguous()
        and c.is_contiguous()
    )


class BF16Plan:
    def __init__(self, a, b, c, *, topk=0, weighted=False, identity=False):
        if not supported(a, b, c, topk):
            raise ValueError("Unsupported BF16 C API tensor contract")
        self.topk, self.weighted = topk, weighted
        self.shapes = (tuple(a.shape), tuple(b.shape), tuple(c.shape))
        self.handle, self.desc, self.moe = P(), P(), P()
        self.layouts, self.closed = [], False
        self.algo = I(0)
        self.binding = None
        self.identity = None
        try:
            _call("HandleCreate", C.byref(self.handle))
            _call("CreateDesc", C.byref(self.desc))
            _attr("DescSetAttribute", self.desc, 0, I(3))
            # AvBv with all-one FP32 scales: no activation/weight quantization.
            self.scale_a = _ones(a.shape[0] * max(topk, 1), a.device)
            self.scale_b = _ones(b.shape[-2] * (b.shape[0] if topk else 1), a.device)
            for index, value in [
                (8, I(6)),
                (1, I(3)),
                (4, I(3)),
                (2, _ptr(self.scale_a)),
                (5, _ptr(self.scale_b)),
            ]:
                _attr("DescSetAttribute", self.desc, index, value)
            if topk:
                _attr("DescSetAttribute", self.desc, 13, C.c_bool(weighted))
            for rows, cols, ld, order, batch in [
                (a.shape[0], a.shape[1], a.shape[1], 1, 1),
                (b.shape[-1], b.shape[-2], b.shape[-1], 0, b.shape[0] if topk else 1),
                (c.numel() // b.shape[-2], b.shape[-2], b.shape[-2], 1, 1),
            ]:
                layout = P()
                _call("MatrixLayoutCreate", C.byref(layout), 2, rows, cols, ld)
                self.layouts.append(layout)
                _attr("MatrixLayoutSetAttribute", layout, 1, I(order))
                _attr("MatrixLayoutSetAttribute", layout, 5, I(batch))
            self.block_m = None
            if topk:
                block = I()
                _call(
                    "FuseMoeGemmGetBlocksizeM",
                    self.handle,
                    self.desc,
                    *self.layouts,
                    topk,
                    C.byref(self.algo),
                    C.byref(block),
                )
                self.block_m = block.value
                _call(
                    "FuseMoeDescCreate",
                    C.byref(self.moe),
                    P(),
                    P(),
                    P(),
                    P(),
                    0,
                    topk,
                    weighted,
                )
                if identity:
                    key = (a.device.index, a.shape[0], self.block_m)
                    if key not in _IDENTITIES:
                        count = math.ceil(a.shape[0] / self.block_m) * self.block_m
                        ids = torch.arange(count, device=a.device, dtype=torch.int32)
                        ids.clamp_max_(a.shape[0])
                        experts = torch.zeros(
                            count // self.block_m, device=a.device, dtype=torch.int32
                        )
                        post = torch.tensor([count], device=a.device, dtype=torch.int32)
                        _IDENTITIES[key] = (ids, experts, post)
                    self.identity = _IDENTITIES[key]
        except Exception:
            self.close()
            raise

    def _bind(self, assignment, weights):
        ids, experts, post = assignment
        addresses = (
            ids.data_ptr(),
            experts.data_ptr(),
            post.data_ptr(),
            weights.data_ptr() if weights is not None else 0,
            ids.numel(),
        )
        if addresses != self.binding:
            for index, tensor in [(0, weights), (1, ids), (2, experts), (3, post)]:
                _attr("FuseMoeDescSetAttribute", self.moe, index, _ptr(tensor))
            _attr("FuseMoeDescSetAttribute", self.moe, 4, I(ids.numel()))
            self.binding = addresses

    def run(self, a, b, c, assignment=None, weights=None):
        if self.topk:
            assignment = self.identity if assignment is None else assignment
            if assignment is None:
                raise ValueError("Routed GEMM requires its queried-block-M assignment")
            self._bind(assignment, weights)
        la, lb, lc = self.layouts
        args = [self.handle, self.desc, _ptr(a), la, _ptr(b), lb, _ptr(c), lc]
        if self.topk:
            args.append(self.moe)
        args.extend(
            [C.byref(self.algo), P(), 0, P(torch.cuda.current_stream().cuda_stream)]
        )
        _call("FuseMoeGemmBasic" if self.topk else "UniversalGemmBasic", *args)
        _STATS["moe" if self.topk else "linear"] += 1
        return c

    def close(self):
        if self.closed:
            return
        if self.moe.value:
            _call("FuseMoeDescDestroy", self.moe)
        for layout in self.layouts:
            _call("MatrixLayoutDestroy", layout)
        if self.desc.value:
            _call("DestroyDesc", self.desc)
        if self.handle.value:
            _call("HandleDestroy", self.handle)
        self.closed = True


def get_plan(a, b, c, *, topk=0, weighted=False, identity=False):
    if not supported(a, b, c, topk):
        return None
    key = (
        a.device.index,
        tuple(a.shape),
        tuple(b.shape),
        tuple(c.shape),
        b.data_ptr(),
        topk,
        weighted,
        identity,
    )
    plan = _PLANS.get(key)
    if plan is not None:
        _STATS["cache_hit"] += 1
        return plan
    if torch.cuda.is_current_stream_capturing():
        _STATS["capture_miss"] += 1
        return None
    plan = BF16Plan(a, b, c, topk=topk, weighted=weighted, identity=identity)
    _PLANS[key] = plan
    _STATS["prepare"] += 1
    return plan


def linear(a, weight, *, identity=False):
    output = torch.empty((a.shape[0], weight.shape[0]), device=a.device, dtype=a.dtype)
    w = weight.unsqueeze(0) if identity else weight
    plan = get_plan(a, w, output, topk=1 if identity else 0, identity=identity)
    if plan is None:
        return None
    return plan.run(a, w, output)

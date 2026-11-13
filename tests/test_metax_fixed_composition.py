# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Numerical and changed-input graph checks for the fixed C550 composition."""

import importlib

import pytest
import torch
import torch.nn.functional as F

import flaggems_vllm

pytestmark = pytest.mark.skipif(
    flaggems_vllm.runtime.device.vendor_name != "metax", reason="MetaX C550 API"
)


def op(name):
    return importlib.import_module("flaggems_vllm.runtime.backend._metax.ops." + name)


def close(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
    assert (error / scale).item() < 0.01
    assert torch.isfinite(actual).all().item()


@pytest.mark.parametrize("rows,vocab", [(1, 4096), (7, 248320), (64, 248320)])
def test_top20_filter_and_race(rows, vocab):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    torch.manual_seed(123)
    x = torch.randn(rows, vocab, device="cuda", dtype=torch.float32)
    p = torch.full((rows,), 0.95, device="cuda")
    k = torch.full((rows,), 20, device="cuda", dtype=torch.int32)
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = op("sampling_top20").apply_top20_top_p(x.clone(), p)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    probs = actual.softmax(-1)
    noise = torch.empty_like(probs).exponential_()
    reference = (probs / noise).argmax(-1)
    torch.testing.assert_close(op("sampling_race").sample(probs, noise), reference)
    for _ in range(2):
        op("sampling_race").sample(probs, noise)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op("sampling_race").sample(probs, noise)
    noise.exponential_()
    graph.replay()
    torch.testing.assert_close(output, (probs / noise).argmax(-1))


@pytest.mark.parametrize("rows", [1, 7, 64, 2048])
def test_hc_bf16_materialization(rows):
    hc = op("hyperconnection_bf16")
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.1
    grouped = x.float().view(rows, 4, 2560)
    reference = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
    reference = (reference * (1 + weight.float().view(1, 4, 2560))).flatten(1)
    normed = hc.grouped_norm(x, weight, 4)
    close(normed, reference.to(x.dtype))
    logits = torch.randn_like(x)
    reduced = (logits.float().sigmoid().to(x.dtype) * normed).view(rows, 4, 2560)
    close(hc.gate_reduce(logits, normed, 4), (reduced.float().mean(1)).to(x.dtype))
    gate = torch.randn(rows, 4, device="cuda", dtype=x.dtype)
    block = torch.randn(rows, 2560, device="cuda", dtype=x.dtype)
    weights = (2 * (gate / 4).sigmoid()).to(x.dtype)
    expected = x.view(rows, 4, 2560) + block[:, None, :] * weights[:, :, None]
    close(hc.inject(gate, block, x, 4), expected.flatten(1))
    down = torch.randn(rows, 320, device="cuda", dtype=x.dtype)
    close(hc.divide_silu_bf16(down, 4), F.silu(down / 4))


@pytest.mark.parametrize("identity", [False, True])
def test_bf16_gemm_changed_input_graph(identity):
    gemm = op("mctlass_bf16")
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(320, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
    for _ in range(2):
        output = gemm.linear(a, w, identity=identity)
    assert output is not None
    close(output, (a.float() @ w.float().T).to(a.dtype))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gemm.linear(a, w, identity=identity)
    for _ in range(2):
        a.normal_()
        graph.replay()
        close(output, (a.float() @ w.float().T).to(a.dtype))


def test_row_gather_masks_and_graph():
    gather = op("host_embedding").gather_host_rows
    table = torch.randn(23, 320, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([[2, -1, 24], [5, 0, 11]], device="cuda", dtype=torch.int64)
    output = torch.empty(6, 320, device="cuda", dtype=table.dtype)
    for _ in range(2):
        gather(table, ids, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather(table, ids, output)
    ids.copy_(torch.tensor([[3, 0, -1], [8, 22, 99]], device="cuda"))
    graph.replay()
    flat = ids.flatten()
    reference = table[flat.clamp(0, 22)].clone()
    reference[(flat < 0) | (flat >= 23)] = 0
    torch.testing.assert_close(output, reference, rtol=0, atol=0)

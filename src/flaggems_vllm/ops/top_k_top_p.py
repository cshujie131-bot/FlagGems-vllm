# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic full-vocabulary top-k/top-p filtering (no RNG).

Reference contract follows vLLM 0.24.0 topk_topp_sampler.py (Apache-2.0).
"""
import torch


def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if k is None and p is None:
        return logits
    values, indices = logits.sort(dim=-1, descending=False)
    if k is not None:
        threshold = values.size(1) - k.to(torch.long)
        threshold = values.gather(1, threshold.unsqueeze(dim=1))
        values.masked_fill_(values < threshold, -float("inf"))
    if p is not None:
        probabilities = values.softmax(dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1, out=probabilities)
        mask = cumulative <= 1 - p.unsqueeze(dim=1)
        mask[:, -1] = False
        values.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)

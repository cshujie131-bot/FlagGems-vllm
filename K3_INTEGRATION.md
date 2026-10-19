# K3 validated-base integration

Base: 3e9e47c988bf834204b416d2a0b3db8c91e7a4de.

Backported public review branches, without upgrading unrelated operators:

- [#950](https://github.com/flagos-ai/FlagGems-vllm/pull/950): explicit FP32-to-input sigmoid materialization plus top16 selection. Default native FGV sigmoid semantics remain unchanged.
- [#951](https://github.com/flagos-ai/FlagGems-vllm/pull/951): BF16 SiTU fusion with the accepted FP32 arithmetic and zero-sign boundary.
- [#952](https://github.com/flagos-ai/FlagGems-vllm/pull/952): small-route non-TLE alignment; full public output capacities/tails preserved. The compact entry lets FL retain ownership of framework buffers.
- [#953](https://github.com/flagos-ai/FlagGems-vllm/pull/953): optional dense MLA callable. API compatibility only, not a performance optimization. Native default kernels and launch settings remain unchanged.

These PRs are unmerged at integration time. Operator test/benchmark selection
is backported without unrelated newer-model mappings. TP8 packing, interleaved
KV layout, staged attention, expert splitting and accepted KDA compatibility
belong to FL; this library does not import plugin-FL or contain a K3 source tree.

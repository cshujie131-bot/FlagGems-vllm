# DSV4 C550 deployment dependency

Base: upstream 5f4dc53cc31b99e83967a2b19195ab1a295aeef6.
This deployment branch deliberately avoids upgrading unrelated operators.

## Operator contributions / attribution

- [#946](https://github.com/flagos-ai/FlagGems-vllm/pull/946),
  a7326fe99f53bb2c67216f3b49b081b8199693b4:
  MetaX non-TLE MoE alignment. The backend implementation is retained unchanged.
  The generic optional-workspace interface is backported to preserve its fallback.
- [#947](https://github.com/flagos-ai/FlagGems-vllm/pull/947),
  7bc8ce9d37f326126bba525239c1bbffc6a7fbb6:
  packed-page gather and corrected launch geometry / metadata-only tuning keys.
  Includes the native implementation/config prerequisites from #916 and the
  missing-config fallback from #933.
- [#755](https://github.com/flagos-ai/FlagGems-vllm/pull/755),
  3a6ae4695cb63c8fe0a39774bf470e580fbbcd10:
  MetaX mhc_pre only. The original contribution and copyright are retained.
  mhc_post is not changed.

## Separately retained compatibility

Dynamic INT8 vendor-rounding compatibility, the existing SiLU codegen choice,
metadata-only launch caching, independent FGV cache directory, and removal of
GDN's import-time process-wide compiler mutation are preserved as source changes.
These are NOT represented as new operator-performance PRs.

TP8 head padding/blockdiag, prefill headpack and staged attention are owned by FL;
the two native MLA operator files are unchanged from the base.
FL owns the graph boundaries. No private source patch installer is required.

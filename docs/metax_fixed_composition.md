# Fixed MetaX C550 serving composition

This revision publishes the operator implementations used by the Qwen3.8
Flash Next BF16 TP8 profile with vLLM 0.24.0 and FlagTree 0.6.1+metax3.6.
It preserves the validated arithmetic, BF16 materialization boundaries and
caller-selected launch parameters. It is a MetaX release branch, not a portable
replacement for the library's default operators.

The matching plugin installation guide and dependency lock are maintained in
[vllm-plugin-FL #635](https://github.com/flagos-ai/vllm-plugin-FL/pull/635).
Use the exact commits in that lock and the matching MetaX Torch/MACA runtime.

| Module below `flaggems_vllm.runtime.backend._metax.ops` | Supported interface |
|---|---|
| `sampling_top20` | `apply_top20_top_p(logits, p)` modifies contiguous FP32 logits in place; the plugin selects Top20 rows and supplies per-row top-p |
| `sampling_race` | `sample(probs, noise, tile, warps)` returns int64 token IDs using the caller's full-vocabulary positive noise; it does not generate or reseed RNG |
| `hyperconnection_bf16` | `grouped_norm`, `gate_reduce`, `inject`, `divide_silu_bf16` preserve BF16 boundaries and require contiguous BF16 tensors |
| `mctlass_bf16` | `get_plan` and `linear` use the matching vendor `libmctlassEx.so`; descriptors/scales are prepared before capture and inputs remain caller-owned |
| `host_embedding` | `gather_host_rows` reads device-addressable storage, zeros invalid rows and writes the caller's output; registration/lifetime belong to the caller |
| `radix_topk` | Row TopK implementation with caller-selected parameters |
| `rotary_bf16`, `qsa_packed` | BF16 rotary/cache-store and sparse attention components with caller-owned buffers |
| `qsa_indexer_fast`, `qsa_paged_native` | Packed indexer and native MACA paged-attention entry points, also exposed through `ops.qwen4` wrappers |

These vendor module interfaces are part of this fixed revision's supported
contract. Import them only after selecting MetaX. Runtime/backend layout is kept
unchanged to avoid an additional forwarding layer. The plugin owns model shape
selection, request metadata and workspace lifetime. Unsupported shapes continue
through the existing caller selection; execution failures are propagated.

Existing top-k/top-p softmax/cumsum stages and initialization of vendor scales
are retained. This release does not claim an entirely Triton-only chain or
introduce new compute fallbacks. It does not modify autotune candidates, promise
training support, or add NVIDIA support for MetaX-specific interfaces.

Run the focused library regression with the matching runtime:

```sh
python -m pytest -q tests/test_metax_fixed_composition.py tests/test_gdn_chunk.py
```

The consuming plugin additionally exercises full QSA, mapped-host storage,
recurrent state and changed-input graph replay. Performance measurements and
experiment traces are not shipped in this guide.

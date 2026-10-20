# MiniMax vLLM 0.24 integration
Base: flagos-ai/FlagGems-vllm b5c2c2249319277ccda4d41aa0f65164bd9b2a19.
Selective source backports (not a whole-library upgrade):
- https://github.com/flagos-ai/FlagGems-vllm/pull/960, 9db6e39bb8fc43f3fe616ec170ded7013b1d9657 (not merged): router BF16-load extension plus its public FP32 router prerequisite from fa78ddee.
- https://github.com/flagos-ai/FlagGems-vllm/pull/961, 0f8f1a788e8976339597674a0f5f0dbc709f4dc8 (not merged): explicit staged norm/RoPE, division activation and ordered reduction modes.
The public MiniMax cache-writer prerequisite and clamped-activation API come from fa78ddee4c3e78ce5047213ff0ef02acff34b2d5. Only the activation enum/helper/API is backported from the newer fused_moe module; existing expert GEMM, quantization and launch code stay on b5c2c224.
No model/TP policy, source-import machinery or private snapshots are added. FL owns headpack, sparse specialization, model shapes and guards. Existing no-TLE alignment, grouped_topk, rotary and other kernels are reused from the pinned base.

Cache isolation: https://github.com/flagos-ai/FlagGems-vllm/pull/962, d1ea137a3b2d790968c5e81b7ad2be278f66e5ee (not merged), preserves the legacy environment fallback.

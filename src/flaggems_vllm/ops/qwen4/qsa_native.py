# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX native attention over caller-owned, prepared virtual-page metadata."""


def native_qsa_attention(q, k, v, out, gate, workspace):
    from flaggems_vllm.runtime.backend._metax.ops.qsa_paged_native import (
        native_qsa_attention as implementation,
    )

    return implementation(q, k, v, out, gate, workspace)

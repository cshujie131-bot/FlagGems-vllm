# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MetaX packed-indexer API over caller-prepared same-request groups."""


def qsa_indexer_fast(
    q,
    k,
    table,
    mapping,
    positions,
    lengths,
    num_columns,
    pack_rows=None,
    *,
    decode_config=None,
    block_n=None,
    num_warps=None,
    num_stages=1,
):
    from flaggems_vllm.runtime.backend._metax.ops.qsa_indexer_fast import (
        qsa_indexer_fast as implementation,
    )

    return implementation(
        q,
        k,
        table,
        mapping,
        positions,
        lengths,
        num_columns,
        pack_rows,
        decode_config=decode_config,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )

# SPDX-License-Identifier: Apache-2.0
"""Row gather from device-addressable host storage into a caller-owned GPU buffer."""

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["N"])
def _gather_rows(
    TABLE,
    IDS,
    OUT,
    N,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    FIRST: tl.constexpr,
    COLS: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BC)
    index_pos = (row // COLS) * INDEX_STRIDE + FIRST + row % COLS
    index = tl.load(IDS + index_pos, row < N, 0).to(tl.int64) - ROW_OFFSET
    valid = (row < N) & (index >= 0) & (index < ROWS)
    # Explicit int64 multiplication is necessary for tables larger than 2GiB.
    value = tl.load(
        TABLE + index[:, None] * H + col[None, :],
        valid[:, None] & (col[None, :] < H),
        0,
    )
    tl.store(
        OUT + row[:, None].to(tl.int64) * H + col[None, :],
        value,
        (row[:, None] < N) & (col[None, :] < H),
    )


def gather_host_rows(
    table,
    indices,
    out,
    *,
    block_rows=8,
    num_warps=4,
    first_column=0,
    num_columns=None,
    row_offset=0,
):
    if table.ndim != 2 or indices.dtype != torch.int64:
        raise ValueError("Host row gather requires a matrix and int64 row IDs")
    if any(
        t.device != table.device or t.device.type != "cuda" or not t.is_contiguous()
        for t in (table, indices, out)
    ):
        raise ValueError("Host row gather tensors must be contiguous on one GPU")
    columns = indices.shape[-1] if indices.ndim == 2 else 1
    selected = columns if num_columns is None else num_columns
    if (
        indices.ndim not in (1, 2)
        or not 0 <= first_column < columns
        or not 1 <= selected <= columns - first_column
    ):
        raise ValueError("Invalid row-ID column selection")
    count = (indices.numel() // columns) * selected
    if out.dtype != table.dtype or out.numel() != count * table.shape[1]:
        raise ValueError("Host row gather output shape/dtype mismatch")
    if count:
        _gather_rows[(tr.cdiv(count, block_rows),)](
            table,
            indices,
            out,
            count,
            table.shape[1],
            table.shape[0],
            columns,
            first_column,
            selected,
            row_offset,
            block_rows,
            tr.next_power_of_2(table.shape[1]),
            num_warps=num_warps,
        )
    return out

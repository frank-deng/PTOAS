# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib templates for ``pto.tscatter``."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES
from ._row_arg import _scalar_literal


def _init_ub_buffer(dst: pto.Tile):
    dtype = dst.dtype
    tile_rows, tile_cols = dst.shape
    lanes = pto.elements_per_vreg(dtype)
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    mask, _ = pto.make_mask(dtype, lanes)
    for row in range(0, tile_rows, 1):
        for col in range(0, tile_cols, lanes):
            pto.vsts(zeros, dst[row, col:], mask)


_SCATTER_MASK_DTYPES = [(dtype, dtype) for dtype in NUMERIC_DTYPES]

@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_index",
    dtypes=[(dtype, dtype, dtype_idx) for dtype in NUMERIC_DTYPES for dtype_idx in ('i32', 'ui32')],
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
)
def template_tscatter_index(src: pto.Tile, dst: pto.Tile, idx: pto.Tile):
    _init_ub_buffer(dst)
    dtype = dst.dtype
    idx_dtype = idx.dtype
    dst_ptr = dst.as_ptr()
    valid_rows, valid_cols = dst.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(idx_dtype, remained)
            src_reg = pto.vlds(src[row, col:])
            idx_reg = pto.vlds(idx[row, col:])
            pto.vscatter(src_reg, dst_ptr, idx_reg, mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p0101",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p0101"),
)
def template_tscatter_mask_row_p0101(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 2
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            dst_reg0, dst_reg1 = pto.vintlv(zeros, src_reg)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p1010",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p1010"),
)
def template_tscatter_mask_row_p1010(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 2
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            dst_reg0, dst_reg1 = pto.vintlv(src_reg, zeros)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p0001",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p0001"),
)
def template_tscatter_mask_row_p0001(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 4
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            tmp_reg0, tmp_reg1 = pto.vintlv(zeros, src_reg)
            dst_reg0, dst_reg1 = pto.vintlv(zeros, tmp_reg0)
            dst_reg2, dst_reg3 = pto.vintlv(zeros, tmp_reg1)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 2:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 3:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p0010",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p0010"),
)
def template_tscatter_mask_row_p0010(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 4
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            tmp_reg0, tmp_reg1 = pto.vintlv(src_reg, zeros)
            dst_reg0, dst_reg1 = pto.vintlv(zeros, tmp_reg0)
            dst_reg2, dst_reg3 = pto.vintlv(zeros, tmp_reg1)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 2:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 3:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p0100",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p0100"),
)
def template_tscatter_mask_row_p0100(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 4
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            tmp_reg0, tmp_reg1 = pto.vintlv(zeros, src_reg)
            dst_reg0, dst_reg1 = pto.vintlv(tmp_reg0, zeros)
            dst_reg2, dst_reg3 = pto.vintlv(tmp_reg1, zeros)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 2:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 3:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p1000",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p1000"),
)
def template_tscatter_mask_row_p1000(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    times = 4
    dst_valid_col = valid_cols * times
    zeros = pto.vbr(_scalar_literal(dtype, 0))
    for row in range(0, valid_rows, 1):
        remained = dst_valid_col
        for col in range(0, valid_cols, lanes):
            src_reg = pto.vlds(src[row, col:])
            tmp_reg0, tmp_reg1 = pto.vintlv(src_reg, zeros)
            dst_reg0, dst_reg1 = pto.vintlv(tmp_reg0, zeros)
            dst_reg2, dst_reg3 = pto.vintlv(tmp_reg1, zeros)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg0, dst[row, col * times:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 2:], mask)
            mask, remained = pto.make_mask(dtype, remained)
            pto.vsts(dst_reg1, dst[row, col * times + lanes * 3:], mask)


@tilelib.tile_template(
    op="pto.tscatter",
    target="a5",
    name="template_tscatter_mask_row_p1111",
    dtypes=_SCATTER_MASK_DTYPES,
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False,
    tags=("scatter", "mask", "row", "p1111"),
)
def template_tscatter_mask_row_p1111(src: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows, valid_cols = src.valid_shape
    lanes = pto.elements_per_vreg(dtype)
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, lanes):
            mask, remained = pto.make_mask(dtype, remained)
            data = pto.vlds(src[row, col:])
            pto.vsts(data, dst[row, col:], mask)

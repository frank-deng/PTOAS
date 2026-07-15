# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL TileLib templates for pto.tgather."""

from ptodsl import pto
import ptodsl.tilelib as tilelib
from ._common import NUMERIC_DTYPES
from ._common import element_store_dist


def gather_dtype_signatures(dtypes=NUMERIC_DTYPES):
    res = []
    for dtype in dtypes:
        for dtype_indices in ('i16', 'ui16', "f16", "bf16", "i32", "ui32", 'f32'):
            res.append((dtype, dtype, dtype_indices))
    return res


@tilelib.tile_template(
    op="pto.tgather",
    target="a5",
    name="template_tgather",
    dtypes=gather_dtype_signatures(),
    iteration_axis="none",
    op_engine="vector",
    op_class="other",
    layouts=("row_major",),
    loop_depth=2,
    is_post_update=False
)
def template_tgather(
    src: pto.Tile,
    dst: pto.Tile,
    indices: pto.Tile):
    dtype = dst.element_type
    dtype_indices = indices.element_type
    elem_bytes = pto.bytewidth(dtype)
    elem_bytes_s1 = pto.bytewidth(dtype_indices)
    valid_rows, valid_cols = indices.valid_shape
    src_ptr = src.as_ptr()
    if pto.const_expr(elem_bytes == 2 and elem_bytes_s1 == 4):
        indices_type = pto.ui32
        lanes = pto.elements_per_vreg(indices_type)
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                indices_reg = pto.vlds(indices[row, col:])
                mask, remained = pto.make_mask(indices_type, remained)
                dst_reg = pto.vgather2_bc(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
                pto.vsts(dst_reg, dst[row, col:], mask)
    elif pto.const_expr(elem_bytes == 1):
        packed_lanes = pto.elements_per_vreg(dtype) >> 1
        result_elem = pto.ui16 if str(dtype) in ("ui8",) else pto.i16
        result_ty = pto.vreg_type(packed_lanes, result_elem)
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, packed_lanes):
                indices_reg = pto.vlds(indices[row, col:])
                mask, remained = pto.make_mask(result_elem, remained)
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, pto.ui16), mask, result_vreg_type=result_ty)
                pto.vsts(dst_reg, dst[row, col:], mask, dist=pto.VStoreDist.PK_B16)
    elif pto.const_expr(elem_bytes == 4):
        indices_type = pto.ui32
        lanes = pto.elements_per_vreg(indices_type)
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                indices_reg = pto.vlds(indices[row, col:])
                mask, remained = pto.make_mask(indices_type, remained)
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
                pto.vsts(dst_reg, dst[row, col:], mask)
    else:
        indices_type = pto.ui16
        lanes = pto.elements_per_vreg(indices_type)
        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                indices_reg = pto.vlds(indices[row, col:])
                mask, remained = pto.make_mask(indices_type, remained)
                dst_reg = pto.vgather2(src_ptr, pto.vbitcast(indices_reg, indices_type), mask)
                pto.vsts(dst_reg, dst[row, col:], mask)


#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# PTODSL rewrite of test/tilelang_st/npu/a5/src/st/testcase/tgather.


from pathlib import Path
import sys
import zlib

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import auto_main, golden_output_case
from ptodsl import pto


PTO_TO_NP_DTYPE = {
    pto.f32: np.float32,
    pto.f16: np.float16,
    pto.i32: np.int32,
    pto.i16: np.int16,
    pto.i8: np.int8,
    pto.ui16: np.uint16,
    pto.ui8: np.uint8,
}


def npy_dtype(pto_type) -> np.dtype:
    return PTO_TO_NP_DTYPE[pto_type]


# Each case is (name, src_dtype, idx_dtype, src_shape, dst_shape).
# The indices tile has the same shape as dst and contains element
# indices into the flat src tile.
CASE_SHAPES = [
    ("f32_1x128_1x64", pto.f32, pto.i32, (1, 128), (1, 64)),
    ("f32_1x64_1x32", pto.f32, pto.i32, (1, 64), (1, 32)),
    ("f32_i32_32x1024_16x64", pto.f32, pto.i32, (32, 1024), (16, 64)),
    ("i32_i32_32x512_16x256", pto.i32, pto.i32, (32, 512), (16, 256)),
    ("f16_i16_16x1024_16x128", pto.f16, pto.i16, (16, 1024), (16, 128)),
    ("i16_i16_32x256_32x64", pto.i16, pto.i16, (32, 256), (32, 64)),
    ("i8_i16_16x128_16x64", pto.i8, pto.i16, (16, 128), (16, 64)),
    ("i8_ui16_16x128_16x64", pto.i8, pto.ui16, (16, 128), (16, 64)),
    ("ui8_ui16_16x128_16x64", pto.ui8, pto.ui16, (16, 128), (16, 64)),
]


def _tgather_body(
    src_ptr,
    offset_ptr,
    dst_ptr,
    *,
    src_rows,
    src_cols,
    dst_rows,
    dst_cols,
    src_dtype,
    idx_dtype,
):
    """Shared kernel body for the tgather cases.

    Loads *src* and *offset* tiles from GM, performs gather using
    ``pto.tile.gather``, and stores *dst* back to GM.
    """

    src_view = pto.make_tensor_view(
        src_ptr, shape=[src_rows, src_cols], strides=[src_cols, 1]
    )
    indices_view = pto.make_tensor_view(
        offset_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1]
    )
    dst_view = pto.make_tensor_view(
        dst_ptr, shape=[dst_rows, dst_cols], strides=[dst_cols, 1]
    )

    src_tile = pto.alloc_tile(shape=[src_rows, src_cols], dtype=src_dtype)
    indices_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=idx_dtype)
    dst_tile = pto.alloc_tile(shape=[dst_rows, dst_cols], dtype=src_dtype)

    pto.tile.load(src_view, src_tile)
    pto.tile.load(indices_view, indices_tile)
    pto.tile.gather(src_tile, dst_tile, indices=indices_tile)
    pto.tile.store(dst_tile, dst_view)


# One decorated kernel per case, each binding static shapes at definition time.
_tgather_kernels = {}
for _name, _src_dtype, _idx_dtype, _src_shape, _dst_shape in CASE_SHAPES:
    _sr, _sc = _src_shape
    _dr, _dc = _dst_shape

    def _make(
        sr=_sr,
        sc=_sc,
        dr=_dr,
        dc=_dc,
        sdt=_src_dtype,
        idt=_idx_dtype,
        kernel_name=f"tgather_{_name}",
    ):
        @pto.jit(name=kernel_name, target="a5")
        def _kernel(
            src_ptr: pto.ptr(sdt, "gm"),
            offset_ptr: pto.ptr(idt, "gm"),
            dst_ptr: pto.ptr(sdt, "gm"),
        ):
            _tgather_body(
                src_ptr,
                offset_ptr,
                dst_ptr,
                src_rows=sr,
                src_cols=sc,
                dst_rows=dr,
                dst_cols=dc,
                src_dtype=sdt,
                idx_dtype=idt,
            )

        return _kernel

    _tgather_kernels[_name] = _make()


def _make_inputs(name, src_dtype, idx_dtype, src_shape, dst_shape):
    src_np = npy_dtype(src_dtype)
    idx_np = npy_dtype(idx_dtype)
    rng_seed = zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.RandomState(rng_seed)

    if np.issubdtype(src_np, np.floating):
        src = rng.uniform(-10.0, 10.0, size=src_shape).astype(src_np)
    else:
        src = rng.randint(-20, 20, size=src_shape).astype(src_np)

    num_src_elements = int(np.prod(src_shape))
    raw_offsets = rng.randint(0, num_src_elements, size=dst_shape).astype(idx_np)

    return [src, raw_offsets]


def _make_expected(src, offsets):
    flat_src = src.reshape(-1)
    dst = np.empty(offsets.shape, dtype=src.dtype)
    flat_offsets = offsets.ravel()
    flat_dst = dst.ravel()
    for i in range(len(flat_dst)):
        flat_dst[i] = flat_src[int(flat_offsets[i])]
    return dst


CASES = []
for _name, _src_dtype, _idx_dtype, _src_shape, _dst_shape in CASE_SHAPES:
    CASES.append(
        golden_output_case(
            "tgather_" + _name,
            _tgather_kernels[_name],
            inputs=lambda _n=_name, _sd=_src_dtype, _id=_idx_dtype, _ss=_src_shape, _ds=_dst_shape: (
                _make_inputs(_n, _sd, _id, _ss, _ds)
            ),
            expected=_make_expected,
            rtol=1e-6,
            atol=1e-6,
        )
    )


auto_main(globals())

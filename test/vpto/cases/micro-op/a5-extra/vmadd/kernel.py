#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from vmadd kernel.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


ELEMS = 1024
SEED = 29


@pto.jit(
    name="a5_extra_vmadd_kernel",
    target="a5",
    backend="vpto",
    mode="explicit",
    source="kernel.pto",
)
def a5_extra_vmadd_kernel(
    f_acc: pto.ptr(pto.f32, "gm"),
    f_lhs: pto.ptr(pto.f32, "gm"),
    f_rhs: pto.ptr(pto.f32, "gm"),
    out_vmadd: pto.ptr(pto.f32, "gm"),
):
    pass


def make_inputs():
    rng = np.random.default_rng(SEED)
    f_acc = rng.uniform(-2.0, 2.0, size=ELEMS).astype(np.float32)
    f_lhs = rng.uniform(-3.0, 3.0, size=ELEMS).astype(np.float32)
    f_rhs = rng.uniform(-1.0, 1.0, size=ELEMS).astype(np.float32)
    return [f_acc, f_lhs, f_rhs]


def make_expected(f_acc, f_lhs, f_rhs):
    return (f_lhs * f_acc + f_rhs).astype(np.float32)


CASES = [
    golden_output_case(
        "a5_extra_vmadd",
        a5_extra_vmadd_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=2e-4,
        atol=2e-4,
    ),
]


auto_main(globals())

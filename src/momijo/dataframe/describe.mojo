# Project:      Momijo
# Module:       src.momijo.dataframe.describe
# File:         describe.mojo
# Path:         src/momijo/dataframe/describe.mojo
#
# Description:  src.momijo.dataframe.describe — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: percentile_f64


from momijo.dataframe.sorting import argsort_f64
from momijo.extras.stubs import len, return

fn percentile_f64(xs: List[Float64], p: Float64) -> Float64
    if len(xs) == 0:
        return 0.0
    var idx = Int(Float64(len(xs) - 1) * p)
    return xs[argsort_f64(xs, True)[idx]]
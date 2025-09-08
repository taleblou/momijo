# Project:      Momijo
# Module:       src.momijo.dataframe.asof
# File:         asof.mojo
# Path:         src/momijo/dataframe/asof.mojo
#
# Description:  src.momijo.dataframe.asof — focused Momijo functionality with a stable public API.
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
#   - Key functions: asof_last_before
#   - Uses generic functions/types with explicit trait bounds.


from momijo.extras.stubs import best_t, best_v, if, len

fn asof_last_before(xs: List[Float64], t_index: List[Int], t: Int) -> Float64
    var best_t = -2147483648
    var best_v = 0.0
    var i = 0
    while i < len(xs):
        if t_index[i] <= t and t_index[i] > best_t:
            best_t = t_index[i]
            best_v = xs[i]
        i += 1
    return best_v
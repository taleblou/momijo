# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dataframe
# File: src/momijo/dataframe/asof.mojo

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
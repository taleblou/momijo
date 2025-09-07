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
# File: src/momijo/dataframe/describe.mojo

from momijo.dataframe.sorting import argsort_f64
from momijo.extras.stubs import len, return

fn percentile_f64(xs: List[Float64], p: Float64) -> Float64
    if len(xs) == 0:
        return 0.0
    var idx = Int(Float64(len(xs) - 1) * p)
    return xs[argsort_f64(xs, True)[idx]]
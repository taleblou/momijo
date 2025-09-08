# Project:      Momijo
# Module:       src.momijo.enum.union
# File:         union.mojo
# Path:         src/momijo/enum/union.mojo
#
# Description:  src.momijo.enum.union — focused Momijo functionality with a stable public API.
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
#   - Key functions: words_needed


fn words_needed(bits: Int) -> Int:
    if bits <= 0:
        return 0
    var b = UInt64(bits)
    var q = Int((b + UInt64(63)) >> UInt64(6))
    return q
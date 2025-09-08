# Project:      Momijo
# Module:       src.momijo.core.asserts
# File:         asserts.mojo
# Path:         src/momijo/core/asserts.mojo
#
# Description:  src.momijo.core.asserts — focused Momijo functionality with a stable public API.
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
#   - Key functions: require, check_equal_int


fn require(cond: Bool, context: String) -> Bool:
    if not cond:
        print("Requirement failed: " + context)
        return False
    return True
fn check_equal_int(a: Int, b: Int, context: String) -> Bool:
    if a != b:
        print("Check failed (Int ==): " + context + "  got=" + String(a) + " expected=" + String(b))
        return False
    return True
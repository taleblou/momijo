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
# Project: momijo.core
# File: momijo/core/asserts.mojo

# Simple runtime checks that return Bool; avoids raising exceptions.

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

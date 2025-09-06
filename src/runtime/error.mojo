# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.runtime
# File: src/momijo/runtime/error.mojo

# Error handling utilities for runtime.
# Provides a lightweight Error struct and helpers for common cases.

@fieldwise_init
struct Error:
    var code: Int
    var message: String

    fn __init__(out self, code: Int, message: String):
        self.code = code
        self.message = message

    fn summary(self) -> String:
        return String("Error(") + String(self.code) + String(", ") + self.message + String(")")


fn make_error(code: Int, message: String) -> Error:
    return Error(code, message)


fn ok() -> Error:
    return Error(0, String("OK"))


fn is_ok(e: Error) -> Bool:
    return e.code == 0


fn _self_test() -> Bool:
    var e1 = ok()
    var e2 = make_error(1, String("fail"))
    var okf = True
    if not is_ok(e1):
        okf = False
    if is_ok(e2):
        okf = False
    if len(e2.summary()) == 0:
        okf = False
    return okf

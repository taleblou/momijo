# Project:      Momijo
# Module:       src.momijo.runtime.error
# File:         error.mojo
# Path:         src/momijo/runtime/error.mojo
#
# Description:  Runtime facilities: device/context management, error handling,
#               environment queries, and state injection patterns.
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
#   - Structs: Error
#   - Key functions: __init__, summary, __copyinit__, __moveinit__, make_error, ok, is_ok, _self_test


@fieldwise_init
struct Error:
    var code: Int
    var message: String
fn __init__(out self, code: Int, message: String) -> None:
        self.code = code
        self.message = message
fn summary(self) -> String:
        return String("Error(") + String(self.code) + String(", ") + self.message + String(")")
fn __copyinit__(out self, other: Self) -> None:
        self.code = other.code
        self.message = other.message
fn __moveinit__(out self, deinit other: Self) -> None:
        self.code = other.code
        self.message = other.message
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
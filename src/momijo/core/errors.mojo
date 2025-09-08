# Project:      Momijo
# Module:       src.momijo.core.errors
# File:         errors.mojo
# Path:         src/momijo/core/errors.mojo
#
# Description:  src.momijo.core.errors — focused Momijo functionality with a stable public API.
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
#   - Structs: Error
#   - Key functions: __copyinit__, __init__, __init__, is_ok, what, code_name, is_fatal, as_string ...
#   - Static methods present.


struct Error(Copyable, Movable, Defaultable):
    var code: Int
    var message: String
    var module: String
    var has_cause: Bool
    var cause: Error
fn __copyinit__(out self, other: Self) -> None:
        self.code = other.code
        self.message = other.message
        self.module = other.module
        self.has_cause = other.has_cause
        self.cause = other.cause

    # zero-arg ctor for Defaultable
fn __init__(out self) -> None:
        self.code = 0
        self.message = ""
        self.module = ""
        self.has_cause = False
        self.cause = Error.none()

    # main ctor
fn __init__(
        out self,
        code: Int,
        message: String,
        module: String = "",
        has_cause: Bool = False,
        cause: Error = Error.none()
    ):
        self.code = code
        self.message = message
        self.module = module
        self.has_cause = has_cause
        if has_cause:
            self.cause = cause
        else:
            self.cause = Error.none()
fn is_ok(self) -> Bool:
        return self.code == 0
fn what(self) -> String:
        return self.message
fn code_name(self) -> String:
        var c = self.code
        if c == 0:  return "OK"
        if c == 1:  return "INVALID_ARGUMENT"
        if c == 2:  return "NOT_FOUND"
        if c == 3:  return "OUT_OF_MEMORY"
        if c == 4:  return "UNIMPLEMENTED"
        if c == 5:  return "INTERNAL"
        if c == 6:  return "IO"
        if c == 7:  return "PARSE"
        if c == 8:  return "RANGE"
        if c == 9:  return "TYPE"
        if c == 10: return "DEVICE_UNAVAILABLE"
        if c == 11: return "NOT_READY"
        if c == 12: return "TIMEOUT"
        if c == 13: return "OVERFLOW"
        if c == 14: return "UNDERFLOW"
        if c == 15: return "DIVIDE_BY_ZERO"
        if c == 16: return "UNSUPPORTED"
        if c == 17: return "PERMISSION_DENIED"
        return "UNKNOWN"
fn is_fatal(self) -> Bool:
        var c = self.code
        return (c == 3) or (c == 5) or (c == 10) or (c == 17)

    # --- APIs your test calls ---
fn as_string(self) -> String:
        return self.describe()

    @staticmethod
fn from_message(msg: String, module: String = "") -> Error:
        # Code 0 (OK) for pure message
        return Error(0, msg, module)
    # ----------------------------
fn to_string(self) -> String:
        return self.describe()
fn describe(self) -> String:
        var prefix = self.module
        if len(prefix) == 0:
            prefix = "momijo"
        var s = "[" + prefix + "] " + self.code_name() + "(" + String(self.code) + "): " + self.message
        var ch = self.chain_string(6)
        if len(ch) > 0:
            s = s + " | cause: " + ch
        return s
fn with_cause(self, cause: Error) -> Error:
        return Error(code=self.code, message=self.message, module=self.module, has_cause=True, cause=cause)
fn root_cause(self) -> Error:
        var cur = self
        var hops = 0
        while cur.has_cause and (hops < 32):
            cur = cur.cause
            hops += 1
        return cur
fn chain_string(self, max_depth: Int = 8) -> String:
        if not self.has_cause:
            return ""
        var out = ""
        var cur = self.cause
        var hops = 0
        while True:
            var m = cur.module
            if len(m) == 0:
                m = "momijo"
            var seg = "[" + m + "] " + cur.code_name() + "(" + String(cur.code) + "): " + cur.message
            if len(out) == 0:
                out = seg
            else:
                out = out + " -> " + seg
            hops += 1
            if (not cur.has_cause) or (hops >= max_depth):
                break
            cur = cur.cause
        return out
fn with_code(self, new_code: Int) -> Error:
        return Error(code=new_code, message=self.message, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn with_message(self, new_message: String) -> Error:
        return Error(code=self.code, message=new_message, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn prefix_message(self, prefix: String) -> Error:
        var msg = self.message
        if len(prefix) > 0:
            msg = prefix + ": " + msg
        return Error(code=self.code, message=msg, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn suffix_message(self, suffix: String) -> Error:
        var msg = self.message
        if len(suffix) > 0:
            msg = msg + " | " + suffix
        return Error(code=self.code, message=msg, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn with_module(self, new_module: String) -> Error:
        return Error(code=self.code, message=self.message, module=new_module, has_cause=self.has_cause, cause=self.cause)

    @staticmethod
fn none() -> Error:
        return Error(0, "", "")
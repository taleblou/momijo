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
# File: src/momijo/core/error.mojo

from momijo.arrow_core.types import UNKNOWN
from momijo.dataframe.helpers import m
from momijo.tensor.broadcast import valid
from momijo.tensor.errors import OK
from momijo.utils.result import g
from pathlib import Path
from pathlib.path import Path

struct MomijoError(Copyable, Movable):
    var code: Int          # 0 = OK
    var message: String
    var module: String     # e.g., "momijo.core.dtype"
    var has_cause: Bool
    var cause: MomijoError # valid only if has_cause == True
fn __copyinit__(out self, other: Self) -> None:
        self.code = other.code
        self.message = other.message
        self.module = other.module
        self.has_cause = other.has_cause
        self.cause = other.cause
fn __init__(
        out self,
        code: Int = 0,
        message: String = "",
        module: String = "",
        has_cause: Bool = False,
        cause: MomijoError = MomijoError.none()
    ):
        self.code = code
        self.message = message
        self.module = module
        self.has_cause = has_cause
        if has_cause:
            self.cause = cause
        else:
            self.cause = MomijoError.none()

    # -------------------------
    # Basics
    # -------------------------
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

    # -------------------------
    # Cause / chaining
    # -------------------------
fn with_cause(self, cause: MomijoError) -> MomijoError:
        return MomijoError(code=self.code, message=self.message, module=self.module, has_cause=True, cause=cause)
fn root_cause(self) -> MomijoError:
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

    # -------------------------
    # Transformations
    # -------------------------
fn with_code(self, new_code: Int) -> MomijoError:
        return MomijoError(code=new_code, message=self.message, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn with_message(self, new_message: String) -> MomijoError:
        return MomijoError(code=self.code, message=new_message, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn prefix_message(self, prefix: String) -> MomijoError:
        var msg = self.message
        if len(prefix) > 0:
            msg = prefix + ": " + msg
        return MomijoError(code=self.code, message=msg, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn suffix_message(self, suffix: String) -> MomijoError:
        var msg = self.message
        if len(suffix) > 0:
            msg = msg + " | " + suffix
        return MomijoError(code=self.code, message=msg, module=self.module, has_cause=self.has_cause, cause=self.cause)
fn with_module(self, new_module: String) -> MomijoError:
        return MomijoError(code=self.code, message=self.message, module=new_module, has_cause=self.has_cause, cause=self.cause)

    # -------------------------
    # Factories
    # -------------------------
    @staticmethod
fn none() -> MomijoError:
        return MomijoError(code=0, message="", module="")

    @staticmethod
fn invalid_argument(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=1, message=msg, module=module)

    @staticmethod
fn not_found(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=2, message=msg, module=module)

    @staticmethod
fn out_of_memory(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=3, message=msg, module=module)

    @staticmethod
fn unimplemented(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=4, message=msg, module=module)

    @staticmethod
fn internal(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=5, message=msg, module=module)

    @staticmethod
fn io_error(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=6, message=msg, module=module)

    @staticmethod
fn parse_error(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=7, message=msg, module=module)

    @staticmethod
fn range_error(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=8, message=msg, module=module)

    @staticmethod
fn type_error(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=9, message=msg, module=module)

    @staticmethod
fn device_unavailable(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=10, message=msg, module=module)

    @staticmethod
fn not_ready(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=11, message=msg, module=module)

    @staticmethod
fn timeout(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=12, message=msg, module=module)

    @staticmethod
fn overflow(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=13, message=msg, module=module)

    @staticmethod
fn underflow(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=14, message=msg, module=module)

    @staticmethod
fn divide_by_zero(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=15, message=msg, module=module)

    @staticmethod
fn unsupported(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=16, message=msg, module=module)

    @staticmethod
fn permission_denied(msg: String, module: String = "") -> MomijoError:
        return MomijoError(code=17, message=msg, module=module)

struct ErrorBuilder(Copyable, Movable):
    var _err: MomijoError
fn __copyinit__(out self, other: Self) -> None:
        self._err = other._err
fn __init__(out self) -> None:
        self._err = MomijoError.none()
fn code(self, c: Int) -> ErrorBuilder:
        var b = ErrorBuilder()
        b._err = self._err.with_code(c)
        return b
fn message(self, msg: String) -> ErrorBuilder:
        var b = ErrorBuilder()
        b._err = self._err.with_message(msg)
        return b
fn module(self, m: String) -> ErrorBuilder:
        var b = ErrorBuilder()
        b._err = self._err.with_module(m)
        return b
fn cause(self, e: MomijoError) -> ErrorBuilder:
        var b = ErrorBuilder()
        b._err = self._err.with_cause(e)
        return b
fn build(self) -> MomijoError:
        return self._err
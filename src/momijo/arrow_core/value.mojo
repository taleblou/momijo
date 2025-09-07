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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/value.mojo

fn __module_name__() -> String:
    return String("momijo/arrow_core/value.mojo")
fn __self_test__() -> Bool:
    var v = Value()
    v.set_int(42)
    if v.is_null(): return False
    if v.as_int() != 42: return False
    if v.type_name() != String("Int"): return False
    v.set_null()
    if not v.is_null(): return False
    return True

# ---------------- ValueTag (enum-like struct) ----------------
struct ValueTag(ExplicitlyCopyable, Movable):
    var tag: Int32
fn __init__(out self, tag: Int32) -> None:
        self.tag = tag
fn __copyinit__(out self, other: Self) -> None:
        self.tag = other.tag

    @staticmethod
fn INT() -> Self:
        return Self(1)

    @staticmethod
fn FLOAT64() -> Self:
        return Self(2)

    @staticmethod
fn STRING() -> Self:
        return Self(3)

    @staticmethod
fn BOOL() -> Self:
        return Self(4)

    @staticmethod
fn NULL() -> Self:
        return Self(0)

# ---------------- Value ----------------
# Keep Movable (no Sized, no copy required).
struct Value(Movable):
    var tag: ValueTag
    var int_val: Int
    var f64_val: Float64
    var str_val: String
    var bool_val: Bool

    # Default ctor
fn __init__(out self) -> None:
        self.tag = ValueTag.NULL()
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = String("")
        self.bool_val = False

    # Overloaded ctor to set all fields at once (lets us return Self without copying)
fn __init__(out self, tag: ValueTag, int_val: Int, f64_val: Float64, str_val: String, bool_val: Bool) -> None:
        self.tag = tag
        self.int_val = int_val
        self.f64_val = f64_val
        self.str_val = str_val
        self.bool_val = bool_val

    # --- Mutating setters (use 'mut self') ---
fn set_int(mut self, v: Int) -> None:
        self.tag = ValueTag.INT()
        self.int_val = v
        self.f64_val = 0.0
        self.str_val = String("")
        self.bool_val = False
fn set_f64(mut self, v: Float64) -> None:
        self.tag = ValueTag.FLOAT64()
        self.int_val = 0
        self.f64_val = v
        self.str_val = String("")
        self.bool_val = False
fn set_string(mut self, v: String) -> None:
        self.tag = ValueTag.STRING()
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = v
        self.bool_val = False
fn set_bool(mut self, v: Bool) -> None:
        self.tag = ValueTag.BOOL()
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = String("")
        self.bool_val = v
fn set_null(mut self) -> None:
        self.tag = ValueTag.NULL()
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = String("")
        self.bool_val = False

    # --- Static constructors (return a freshly constructed temporary; no copy needed) ---
    @staticmethod
fn from_int(v: Int) -> Self:
        return Self(ValueTag.INT(), v, 0.0, String(""), False)

    @staticmethod
fn from_f64(v: Float64) -> Self:
        return Self(ValueTag.FLOAT64(), 0, v, String(""), False)

    @staticmethod
fn from_string(v: String) -> Self:
        return Self(ValueTag.STRING(), 0, 0.0, v, False)

    @staticmethod
fn from_bool(v: Bool) -> Self:
        return Self(ValueTag.BOOL(), 0, 0.0, String(""), v)

    # ---------- Accessors ----------
fn as_int(self) -> Int:
        if self.tag.tag == ValueTag.INT().tag:
            return self.int_val
        return 0
fn as_f64(self) -> Float64:
        if self.tag.tag == ValueTag.FLOAT64().tag:
            return self.f64_val
        return 0.0
fn as_string(self) -> String:
        if self.tag.tag == ValueTag.STRING().tag:
            return self.str_val
        return String("")
fn as_bool(self) -> Bool:
        if self.tag.tag == ValueTag.BOOL().tag:
            return self.bool_val
        return False
fn is_null(self) -> Bool:
        return self.tag.tag == ValueTag.NULL().tag
fn type_name(self) -> String:
        var t = self.tag.tag
        if t == ValueTag.INT().tag:     return String("Int")
        if t == ValueTag.FLOAT64().tag: return String("Float64")
        if t == ValueTag.STRING().tag:  return String("String")
        if t == ValueTag.BOOL().tag:    return String("Bool")
        return String("Null")
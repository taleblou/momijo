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
# File: src/momijo/arrow_core/string_array.mojo

from momijo.arrow_core.byte_string_array import (

    ByteStringArray,
    byte_string_array_from_strings,
    byte_string_array_from_optional_strings
)
fn __module_name__() -> String:
    return String("momijo/arrow_core/string_array.mojo")
fn __self_test__() -> Bool:
    return True

# --- Module-level helpers ---
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]; idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# --- StringArray wrapper ---
struct StringArray(Copyable, Movable, Sized):
    var inner: ByteStringArray

    # ---------- Constructors ----------
fn __init__(out self) -> None:
        self.inner = ByteStringArray()

    # Use assign_... instead of alt-init with out self
fn assign_from_strings(mut self, strings: List[String]) -> None:
        self.inner = byte_string_array_from_strings(strings)
fn assign_from_optional_strings(mut self, strings: List[Optional[String]]) -> None:
        self.inner = byte_string_array_from_optional_strings(strings)

    # ---------- Properties ----------
    @always_inline
fn __len__(self) -> Int:
        return self.inner.len()
fn len(self) -> Int:
        return self.inner.len()

    # ---------- Access ----------
fn is_valid(self, i: Int) -> Bool:
        return self.inner.is_valid(i)
fn is_null(self, i: Int) -> Bool:
        return self.inner.is_null(i)
fn get(self, i: Int) -> String:
        return self.inner.get(i)
fn get_or(self, i: Int, default: String) -> String:
        return self.inner.get_or(i, default)

    # ---------- Mutation ----------
fn push(mut self, s: String, valid: Bool = True) -> None:
        if valid:
            self.inner.push(s, True)
        else:
            self.inner.push(s, False)
fn push_null(mut self) -> None:
        self.inner.push_null()
fn clear(mut self) -> None:
        self.inner.clear()

    # ---------- Conversion ----------
fn to_strings(self) -> List[String]:
        return self.inner.to_strings()
fn to_optional_strings(self) -> List[Optional[String]]:
        return self.inner.to_optional_strings()

    # ---------- Utility ----------
fn slice(self, start: Int, count: Int) -> StringArray:
        var out = StringArray()
        out.inner = self.inner.slice(start, count)
        return out

# ---------- Public factories for StringArray ----------
fn string_array_from_strings(strings: List[String]) -> StringArray:
    var s = StringArray()
    s.inner = byte_string_array_from_strings(strings)
    return s
fn string_array_from_optional_strings(strings: List[Optional[String]]) -> StringArray:
    var s = StringArray()
    s.inner = byte_string_array_from_optional_strings(strings)
    return s
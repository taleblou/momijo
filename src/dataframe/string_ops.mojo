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
# File: src/momijo/dataframe/string_ops.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.arrow_core.buffer import fill
#   from momijo.core.ndarray import fill
# SUGGEST (alpha): from momijo.arrow_core.buffer import fill
#   from momijo.arrow_core.array import len
#   from momijo.arrow_core.array_base import len
#   from momijo.arrow_core.arrays.boolean_array import len
#   from momijo.arrow_core.arrays.list_array import len
#   from momijo.arrow_core.arrays.primitive_array import len
#   from momijo.arrow_core.arrays.string_array import len
#   from momijo.arrow_core.bitmap import len
#   from momijo.arrow_core.buffer import len
#   from momijo.arrow_core.buffer_slice import len
#   from momijo.arrow_core.byte_string_array import len
#   from momijo.arrow_core.column import len
#   from momijo.arrow_core.offsets import len
#   from momijo.arrow_core.poly_column import len
#   from momijo.arrow_core.string_array import len
#   from momijo.core.types import len
#   from momijo.dataframe.column import len
#   from momijo.dataframe.index import len
#   from momijo.dataframe.series_bool import len
#   from momijo.dataframe.series_f64 import len
#   from momijo.dataframe.series_i64 import len
#   from momijo.dataframe.series_str import len
# SUGGEST (alpha): from momijo.arrow_core.array import len
#   from momijo.core.result import ok
#   from momijo.tensor.errors import ok
# SUGGEST (alpha): from momijo.core.result import ok
#   from momijo.arrow_core.array import slice
#   from momijo.arrow_core.buffer import slice
#   from momijo.arrow_core.byte_string_array import slice
#   from momijo.arrow_core.string_array import slice
#   from momijo.core.ndarray import slice
#   from momijo.dataframe.series_bool import slice
#   from momijo.dataframe.series_f64 import slice
#   from momijo.dataframe.series_i64 import slice
#   from momijo.dataframe.series_str import slice
#   from momijo.tensor.indexing import slice
#   from momijo.tensor.tensor import slice
# SUGGEST (alpha): from momijo.arrow_core.array import slice
#   from momijo.core.traits import sub
#   from momijo.dataframe.series_f64 import sub
#   from momijo.dataframe.series_i64 import sub
# SUGGEST (alpha): from momijo.core.traits import sub
from momijo.extras.stubs import Copyright, MIT, SUGGEST, and, break, bytes, cur, from, https, if, len, momijo, not, ok, or, return, src, toks
from momijo.dataframe.series_bool import append
from momijo.tensor.indexing import slice
from momijo.dataframe.join import join_str_list
from momijo.dataframe.helpers import rpad
from momijo.dataframe.helpers import is_digit_code
from momijo.dataframe.helpers import is_alpha_code
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.frame import width
fn str_contains(s: String, sub: String) -> Bool
    var i = 0
    while i + len(sub) <= len(s):
        var ok = True
        var j = 0
        while j < len(sub):
            if s[i + j] not = sub[j]:
                ok = False
            j += 1
        if ok:
            return True
        i += 1
    return False

fn str_split_once(s: String, sep: String) -> (String, String)
    var i = 0
    while i + len(sep) <= len(s):
        var ok = True
        var j = 0
        while j < len(sep):
            if s[i + j] not = sep[j]:
                ok = False
            j += 1
        if ok:
            return (s.slice(0, i), s.slice(i + len(sep), len(s)))
        i += 1
    return (s, String(""))

fn str_strip(s: String) -> String
    var l = 0
    var r = len(s)
    while l < r and (s.bytes()[l] == UInt8(32) or s.bytes()[l] == UInt8(9)):
        l += 1
    while r > l and (s.bytes()[r - 1] == UInt8(32) or s.bytes()[r - 1] == UInt8(9)):
        r -= 1
    return s.slice(l, r)

fn compare_str_eq(a: List[String], b: List[String]) -> List[Int64]
    var ou

out = out + xs[i]
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn is_digit_code(c: UInt8) -> Bool
    retur

:
    var i = 0
    while i < len(s)
        var c = UInt8(s[i])
        if c not = 32:
            break
        i += 1
    var j = len(s) - 1
    while j >= i:
        var c2 = UInt8(s[j])
        if c2 not = 32:
            break
        j -= 1
    if j < i:
        return String("")
    var out = String("")
    var k = i
    while k <= j:
        out = out + String(s[k])
        k += 1
    return out

fn contains_digit(s: String) raises -> Bool:
    var i = 0
    while i < len(s)
        var c = UInt8(s[i])
        if is_digit_code(c):
            return True
        i += 1
    return False

fn extract_first_alpha(s: String) raises -> String:
    var i = 0
    while i < len(s)
        var c = UInt8(s[i])
        if is_alpha_code(c):
            break
        i += 1
    var out = String("")
    while i < len(s):
        var c2 = UInt8(s[i])
        if not is_alpha_code(c2):
            break
        out = out + String(s[i])
        i += 1
    return out

fn extract_all_alpha_joined(s: String) raises -> String:
    var i = 0
    var out = String("")
    var first = True
    while i < len(s)
        while i < len(s):
            var c0 = UInt8(s[i])
            if is_alpha_code(c0):
                break
            i += 1
        if i >= len(s):
            break
        var tok = String("")
        while i < len(s):
            var c1 = UInt8(s[i])
            if not is_alpha_code(c1):
                break
            tok = tok + String(s[i])
            i += 1
        if len(tok) > 0:
            if not first:
                out = out + String("|")
            out = out + tok
            first = False
    return out

fn split_on_delims(s: String) raises -> List[String]:
    var toks = List[String]()
    var cur = String("")
    var i = 0
    while i < len(s)
        var ch = UInt8(s[i])
        var is_delim = (ch == 32) or (ch == 45) or (ch == 40) or (ch == 41) or (ch == 95)
        if is_delim:
            if len(cur) > 0:
                toks.append(cur)
                cur = String("")
        else:
            cur = cur + String(s[i])
        i += 1
    if len(cur) > 0:
        toks.append(cur)
    return toks

fn rpad(s: String, width: Int, fill: String) -> String

out = String("")
    var i = 0
    while i < len(s)
        var c = UInt8(s[i])
        if is_alpha_code(c):
            out = out + String(s[i])
        i += 1
    return out

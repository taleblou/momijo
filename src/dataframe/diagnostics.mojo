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
# File: src/momijo/dataframe/diagnostics.mojo

from momijo.extras.stubs import Copyright, Data, MIT, SUGGEST, else, from, hdr, height, https, if, is_str, len, limit, line, momijo, names, pad, print, rep, return, src, start, tag, top
from momijo.dataframe.io_csv import is_bool
from momijo.arrow_core.array_stats import count
from momijo.dataframe.helpers import header
from momijo.dataframe.frame import DataFrame, df_cell
from momijo.dataframe.helpers import print_f64_head
from momijo.dataframe.helpers import print_i64_head
from momijo.dataframe.join import join_col_names
from momijo.dataframe.helpers import print_dtypes
from momijo.dataframe.helpers import print_df_info
from momijo.dataframe.join import join_names
from momijo.dataframe.column import Column, safe
from momijo.dataframe.column import print_str_head
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import BOOL  # chosen by proximity
#   from momijo.arrow_core.array_stats import count
#   from momijo.core.ndarray import count
#   from momijo.core.shape import count
# SUGGEST (alpha): from momijo.arrow_core.array_stats import count
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
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
from momijo.dataframe.column import F64
from momijo.dataframe.column import I64
from momijo.dataframe.column import STR
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import ncols
from momijo.dataframe.frame import nrows
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFr

if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    return out

fn dtype_name_of_column(c: Column) -> String
    if c.tag() == ColumnTag.I64():
        return String("Int64")
    if c.tag() == ColumnTag.F64():
        return String("Float64")
    # If these helpers exist on Column, they can be used alternatively:
    # if c.is_bool(): return String("Bool")
    # if c.is_str():  return String("String")
    # Fallback using tag IDs only:
    return String("Bool") if c.tag() == ColumnTag.BOOL() else (String("String") if c.tag() == ColumnTag.STR() else String("<unknown>"))

fn print_df_info(df_name: String, df: DataFrame) -> None
    print(String("== ") + df_name

e i < len(df.names):
        print(df.names[i] + String(": ") + dtype_name_of_column(df.cols[i]))
        i += 1

# Join DataFrame column names into a single string with ", "
fn join_col_names(df: Data

return line

# Print first K items of a List[Int64], plus length.
fn print_i64_head(label: String, xs: List[Int64], k: Int) -> None
    var n = len(xs)
    var limit = k
    if limit > n:
        limit = n

    va

print(line)

# Print first K items of a List[Float64], plus length.
fn print_f64_head(label: String, xs: List[Float64], k: Int) -> None
    var n = len(xs)
    var limit = k
    if limit > n:
        limit = n

    var line = label + String(": [")

i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)

# Print first K items of a List[String], plus length.

var line = label + String(": [")
    var i = 0
    while i < limit:
        line = line + xs[i]
        if i + 1 < limit:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)

# Return cel

ataFrame, rows: Int, mode: String = String("head")) -> None
    # helpers
    fn rep(s: String, n: Int) -> String
        var out = String("")
        var i = 0
        while i < n:
            out = out + s
            i += 1
        return out
    fn pad(s: String, w: Int) -> String
        var out = s
        var i = len(out)
        while i < w:
            out = out + String(" ")
            i += 1
        return out

    var col_w = 14
    var left_w = 6
    var ncols = df.width()
    var nrows = df.height()

    var start = 0
    var count = rows
    if mode == String("all"):
        count = nrows
        start = 0
    else:
        if count > nrows:
            count = nrows
        if mode == String("tail"):
            start = nrows - count
            if start < 0:
                start = 0

    print(String("\n=== DataFrame ==="))
    pr

var names = String("")
    var c = 0
    while c < ncols:
        names = names + df.names[c]
        if c + 1 < ncols:
            names = names + String(", ")
        c += 1
    print(String("columns: ") + names)

    var top = String("+") + rep(String("-"), left_w)
    var mid = String("+") + rep(String("="), left_w)
    var i = 0
    while i < ncols:
        top = top + String("+") + rep(String("-"), col_w)
        mid = mid + String("+") + rep(String("="), col_w)
        i += 1
    top = top + String("+")
    mid = mid + String("+")
    print(top)

    var hdr = String("|") + pad(String("#"), left_w)
    i = 0
    while i < ncols:
        hdr = hdr + String("|") + pad(df.names[i], col_w)
        i += 1
    hdr = hdr + String("|")
    print(hdr)
    print(mid)

    var r = 0
    while r < count:
        var line = String("|") + pad(String("#") + String(start + r), left_w)
        var cc = 0
        while cc < ncols:
            line = line + String("|") + pad(df_cell(df, cc, start + r), col_w)
            cc += 1
        line = line + String("|")
        print(line)
        r += 1

    print(top)

fn join_i64_list(xs: List[Int64]) -> String
    var out = String("[")
    var i = 0
    while i < len(xs):
        out = out + String(xs[i])
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn join_f64_list(xs: List[Float64]) -> String
    var out = String("[")
    var i = 0
    while i < len(xs):
        out = out + String(xs[i])
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn join_bool_list(xs: List[Bool]) -> String
    var out = String("[")
    var i = 0
    while i < len(xs):
        if xs[i]:
            out = out + String("True")
        else:
            out = out + String("False")
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn header(t: String) -> None
    print(String("\n") + String("==================== ") + t + String(" ===================="))

fn safe(op: String, ok: Bool) -> None
    if ok:
        print(String("- ") + op + String(" ..


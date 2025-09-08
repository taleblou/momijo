# Project:      Momijo
# Module:       src.momijo.dataframe.diagnostics
# File:         diagnostics.mojo
# Path:         src/momijo/dataframe/diagnostics.mojo
#
# Description:  src.momijo.dataframe.diagnostics — focused Momijo functionality with a stable public API.
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
#   - Key functions: dtype_name_of_column, print_df_info, join_col_names, print_i64_head, print_f64_head, rep, pad, join_i64_list ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array_stats import count
from momijo.dataframe.column import BOOL, Column, F64, I64, STR, safe
from momijo.dataframe.frame import DataFrame, df_cell, ncols, nrows, width
from momijo.dataframe.helpers import header, print_df_info, print_f64_head, print_i64_head
from momijo.dataframe.join import join_col_names
from momijo.extras.stubs import Data, else, hdr, height, if, len, limit, line, names, pad, print, rep, return, start, tag, top

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
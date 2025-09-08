# Project:      Momijo
# Module:       src.momijo.dataframe.na
# File:         na.mojo
# Path:         src/momijo/dataframe/na.mojo
#
# Description:  src.momijo.dataframe.na — focused Momijo functionality with a stable public API.
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
#   - Key functions: fillna_col_i64, fillna_col_f64, fillna_col_bool, dropna_rows_any, isin_i64, isin_f64
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_bool, col_f64, col_i64, col_str, df_make
from momijo.dataframe.column import BOOL, Column, ColumnTag, F64, I64, get_bool, get_f64, get_i64, name
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import isna_str
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import found, height, if, keep, len, ok, tag, vals

fn fillna_col_i64(c: Column, fill: Int64) -> Column
    var out = List[Int64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.I64():
            out.append(c.get_i64(i))
        else:
            out.append(Int64(0))
        i += 1
    return col_i64(c.name(), out)
fn fillna_col_f64(c: Column, fill: Float64) -> Column
    var out = List[Float64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.F64():
            out.append(c.get_f64(i))
        else:
            out.append(0.0)
        i += 1
    return col_f64(c.name(), out)
fn fillna_col_bool(c: Column, fill: Bool) -> Column
    var out = List[Bool]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.BOOL():
            out.append(c.get_bool(i))
        else:
            out.append(fill)
        i += 1
    return col_bool(c.name(), out)

# Drop any row that has String-NA in any column (demo-level NA semantics)
fn dropna_rows_any(df: DataFrame) -> DataFrame
    var keep = List[Int]()
    var r = 0
    while r < df.height():
        var ok = True
        var c = 0
        while c < df.width():
            var v = df.cols[c].get_string(r)
            if isna_str(v):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
    var names = df.names
    var cols = List[Column]()
    var cc = 0
    while cc < df.width():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc].get_string(keep[i]))
            i += 1
        cols.append(col_str(names[cc], vals))
        cc += 1
    return df_make(names, cols)
fn isin_i64(xs: List[Int64], universe: List[Int64]) -> List[Bool]
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out
fn isin_f64(xs: List[Float64], universe: List[Float64]) -> List[Bool]
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out
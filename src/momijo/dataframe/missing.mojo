# Project:      Momijo
# Module:       dataframe.missing
# File:         missing.mojo
# Path:         dataframe/missing.mojo
#
# Description:  dataframe.missing — Missing module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: —
#   - Key functions: fillna_str_col, dropna_rows, fillna_col_i64, fillna_col_f64, fillna_col_bool, dropna_rows_any, isin_i64, isin_f64, dropna_any

from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import isna_str
from momijo.dataframe.series_bool import append
from momijo.dataframe.api import col_bool, col_f64, col_i64, col_str, df_make
from momijo.dataframe.column import BOOL, Column, ColumnTag, F64, I64, get_bool, get_f64, get_i64, name

fn fillna_str_col(c: Column, fill: String) -> Column
    var out = List[String]()
    var r = 0
    while r < c.len():
        var v = c[r]
        if isna_str(v): out.append(fill)
        else: out.append(v)
        r += 1
    return col_str(String("filled"), out)

# Drop any row that has an NA-like string cell in any column.
fn dropna_rows(df: DataFrame) -> DataFrame
    var keep = List[Int]()
    var r = 0
    while r < df.nrows():
        var ok = True
        var c = 0
        while c < df.ncols():
            if isna_str(df.cols[c][r]):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
# build
    var col_names = df.col_names
    var cols = List[Column]()
    var cc = 0
    while cc < df.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc][keep[i]])
            i += 1
        cols.append(col_str(col_names[cc], vals))
        cc += 1
    return df_make(col_names, cols)


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
    while r < df.nrows():
        var ok = True
        var c = 0
        while c < df.ncols():
            var v = df.cols[c][r]
            if isna_str(v):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
    var col_names = df.col_names
    var cols = List[Column]()
    var cc = 0
    while cc < df.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc][keep[i]])
            i += 1
        cols.append(col_str(col_names[cc], vals))
        cc += 1
    return df_make(col_names, cols)
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

# Remove rows that have NA in any column (facade over missing.dropna_rows)
fn dropna_any(df0: DataFrame) -> DataFrame:
    return _dropna_rows(df0)# Drop duplicate rows based on subset of columns
fn drop_duplicates(df: DataFrame, subset: List[String], keep: String = "first") -> DataFrame:
    var seen = List[List[String]]()
    var new_cols = List[List[String]]()
    var ncols = df.ncols()
    var col_idx = List[Int]()

    var i = 0
    while i < ncols:
        new_cols.append(List[String]())
        i += 1

    var r = 0
    while r < df.nrows():
        var row_vals = List[String]()
        var j = 0
        while j < len(subset):
            var idx = df.find_col(subset[j])
            if idx != -1:
                row_vals.append(df.cols[idx][r])
            j += 1

        var duplicate = False
        var k = 0
        while k < len(seen):
            var match_row = True
            var m = 0
            while m < len(row_vals):
                if seen[k][m] != row_vals[m]:
                    match_row = False
                    break
                m += 1
            if match_row:
                duplicate = True
                break
            k += 1

        if not duplicate or (keep == "last"):
            # Add row to new_cols
            var c = 0
            while c < ncols:
                new_cols[c].append(df.cols[c][r])
                c += 1
            seen.append(row_vals)

        r += 1

    return DataFrame(df.col_names, new_cols, df.index_vals, df.index_name)


# Drop rows with any NA
fn _dropna_rows(df0: DataFrame) -> DataFrame:
    var new_cols = List[List[String]]()
    var ncols = df0.ncols()

    var c = 0
    while c < ncols:
        new_cols.append(List[String]())
        c += 1

    var new_index = List[String]()

    var r = 0
    while r < df0.nrows():
        var has_na = False
        var c2 = 0
        while c2 < ncols:
            if _isna(df0.cols[c2][r]):
                has_na = True
                break
            c2 += 1

        if not has_na:
            c2 = 0
            while c2 < ncols:
                new_cols[c2].append(df0.cols[c2][r])
                c2 += 1
            if len(df0.index_vals) > r:
                new_index.append(df0.index_vals[r])
        r += 1

    return DataFrame(df0.col_names, new_cols, new_index, df0.index_name)

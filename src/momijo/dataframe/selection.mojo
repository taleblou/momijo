# Project:      Momijo
# Module:       dataframe.selection
# File:         selection.mojo
# Path:         dataframe/selection.mojo
#
# Description:  dataframe.selection — Row/column selection and indexing helpers (loc/iloc/select).
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
#   - Structs: RowRange, ColRange, Mask, ColumnBoolOps
#   - Key functions: __init__, _copy_meta, _col_from_list, select, loc, __copyinit__, mask_and, filter_rows, col_ge, _is_digit, col_isin, where, iloc, filter, ge, isin, between, at

from momijo.dataframe.frame import DataFrame as DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.io_bytes import str_to_bytes

struct RowRange:
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop = stop

struct ColRange:
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop = stop

# copy basic metadata (index) from src into out_df (out parameter)
fn _copy_meta(src: DataFrame, out out_df: DataFrame):
    out_df.index_name = String(src.index_name)
    out_df.index_vals = List[String]()
    var i = 0
    while i < len(src.index_vals):
        out_df.index_vals.append(String(src.index_vals[i]))
        i += 1

# helper: build a Column from a list of strings with a name
fn _col_from_list(vals: List[String], name: String) -> Column:
    var s = SeriesStr(vals, name)
    var col = Column()
    col.from_str(s)
    return col

# select columns by names
fn select(df: DataFrame, cols: List[String]) -> DataFrame:
    var out = DataFrame()

# copy index metadata
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var k = 0
    while k < len(df.index_vals):
        out.index_vals.append(String(df.index_vals[k]))
        k += 1

# initialize column containers
    out.col_names = List[String]()
    out.names = List[String]()
    out.cols = List[Column]()

# select requested columns by name
    var i = 0
    while i < len(cols):
        var name = cols[i]

# find the index of this column name in the source DataFrame
        var j = 0
        while j < len(df.col_names) and df.col_names[j] != name:
            j += 1

        if j < len(df.col_names):
            out.col_names.append(String(df.col_names[j]))
            out.names.append(String(df.col_names[j]))
            out.cols.append(df.cols[j])

        i += 1

    return out


# loc: slice rows by inclusive RowRange, and columns by names
fn loc(df: DataFrame, rows: RowRange, cols: List[String]) -> DataFrame:
    var sub = select(df, cols)
    var out = DataFrame() 
    out.col_names = List[String]()
    out.names = List[String]()
    out.cols = List[Column]()
    out.index_name = String(sub.index_name)
    out.index_vals = List[String]()
    var __k = 0
    while __k < len(sub.index_vals):
        out.index_vals.append(String(sub.index_vals[__k]))
        __k += 1

        out.col_names = List[String]()
        out.cols = List[Column]()
        var c = 0
        while c < len(sub.col_names):
            out.col_names.append(String(sub.col_names[c]))
            c += 1
# clamp range
        var start = rows.start
        var stop  = rows.stop
        if start < 0: start = 0
        var nrows = 0
        if len(sub.cols) > 0:
            nrows = sub.cols[0].len()
        if stop >= nrows: stop = nrows - 1
# build columns
        c = 0
        while c < len(sub.col_names):
            var vals = List[String]()
            var r = start
            while r <= stop:
                vals.append(String(sub.cols[c].get_string(r)))
                r += 1
            out.cols.append(_col_from_list(vals, sub.col_names[c]))
            c += 1
    return out

# filter rows by boolean mask

# boolean mask to filter rows
struct Mask:
    var vals: List[Bool]
    fn __init__(out self):
        self.vals = List[Bool]()
    fn __copyinit__(out self, other: Mask):
        self.vals = List[Bool]()
        var i = 0
        while i < len(other.vals):
            self.vals.append(other.vals[i])
            i += 1

fn mask_and(a: Mask, b: Mask) -> Mask:
    var out = Mask()
    var n = len(a.vals)
    var i = 0
    while i < n and i < len(b.vals):
        out.vals.append(a.vals[i] and b.vals[i])
        i += 1
    return out

fn filter_rows(df: DataFrame, mask: List[Bool]) -> DataFrame:
    var out = DataFrame()
    out.col_names = List[String]()
    out.names = List[String]()
    out.cols = List[Column]()
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var __k = 0
    while __k < len(df.index_vals):
        out.index_vals.append(String(df.index_vals[__k]))
        __k += 1

        out.col_names = List[String]()
        out.cols = List[Column]()
        var c = 0
        while c < len(df.col_names):
            out.col_names.append(String(df.col_names[c]))
            var vals = List[String]()
            var i = 0
            var n = df.cols[c].len()
            while i < n and i < len(mask):
                if mask[i]:
                    vals.append(String(df.cols[c].get_string(i)))
                i += 1
            out.cols.append(_col_from_list(vals, df.col_names[c]))
            c += 1
    return out


# build mask: column >= value (numeric compare; non-parsable → False)
fn col_ge(df: DataFrame, name: String, value: Int) -> Mask:
    var out = Mask()

# find column index by name
    var j = 0
    while j < len(df.col_names) and df.col_names[j] != name:
        j += 1

# column not found: all False with correct length
    var r = 0
    if j >= len(df.col_names):
        while r < df.nrows():
            out.vals.append(False)
            r += 1
        return out

# ASCII constants (avoid any indexing that may raise)
    var ZERO: UInt8  = UInt8(48)   # '0'
    var NINE: UInt8  = UInt8(57)   # '9'
    var MINUS: UInt8 = UInt8(45)   # '-'
    var PLUS: UInt8  = UInt8(43)   # '+'

# digit check on UInt8 without any raising ops
    fn _is_digit(b: UInt8) -> Bool:
        return (b >= ZERO) and (b <= NINE)

# iterate rows and parse integer from string cell
    r = 0
    var nrows = df.nrows()
    while r < nrows:
        var s = String(df.cols[j].get_string(r))   # assumes get_string exists
        var bs = str_to_bytes(s)                   # convert once to bytes
        var n = len(bs)

# default: invalid → False
        var bad = (n == 0)
        var sign = 1
        var i = 0

# optional leading sign
        if not bad and (bs[0] == MINUS or bs[0] == PLUS):
            if n == 1:
                bad = True           # lone '+' or '-' is invalid
            else:
                sign = -1 if bs[0] == MINUS else 1
                i = 1

        var acc = 0
        while (not bad) and (i < n):
            var ch = bs[i]
            if not _is_digit(ch):
                bad = True
                break
# acc = acc * 10 + (ch - '0')
            acc = acc * 10 + (Int(ch) - Int(ZERO))
            i += 1

        if not bad:
            acc = acc * sign
            out.vals.append(acc >= value)
        else:
            out.vals.append(False)

        r += 1

    return out


# build mask: column value in set of strings
fn col_isin(df: DataFrame, name: String, values: List[String]) -> Mask:
    var out = Mask()
    var j = 0
    while j < len(df.col_names) and df.col_names[j] != name:
        j += 1
    var r = 0
    if j >= len(df.col_names):
        while r < df.nrows():
            out.vals.append(False)
            r += 1
        return out
    while r < df.nrows():
        var s = String(df.cols[j].get_string(r))
        var k = 0
        var found = False
        while k < len(values):
            if s == values[k]:
                found = True
                break
            k += 1
        out.vals.append(found)
        r += 1
    return out

# where: return only rows where mask is True
fn where(df: DataFrame, mask: Mask) -> DataFrame:
    var bools = List[Bool]()
    var i = 0
    while i < len(mask.vals):
        bools.append(mask.vals[i])
        i += 1
    return filter_rows(df, bools)

# iloc: select by explicit row indices and a half-open column range [start, stop)
fn iloc(df: DataFrame, row_indices: List[Int], col_range: ColRange) -> DataFrame:
 
    var out = DataFrame()
    out.col_names = List[String]()
    out.names = List[String]()
    out.cols = List[Column]()
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var __k = 0
    while __k < len(df.index_vals):
        out.index_vals.append(String(df.index_vals[__k]))
        __k += 1

        out.col_names = List[String]()
        out.cols = List[Column]()
        var c = col_range.start
        while c < col_range.stop and c < df.ncols():
            out.col_names.append(String(df.col_names[c]))
            var vals = List[String]()
            var i = 0
            while i < len(row_indices):
                var r = row_indices[i]
                if r >= 0 and r < df.nrows():
                    vals.append(String(df.cols[c].get_string(r)))
                i += 1
            out.cols.append(_col_from_list(vals, df.col_names[c]))
            c += 1
    return out

# Public alias expected by All.mojo
fn filter(df: DataFrame, mask: List[Bool]) -> DataFrame:
    return filter_rows(df, mask)

# Overload: accept Mask directly (for All.mojo compatibility)
fn filter(df: DataFrame, mask: Mask) -> DataFrame:
    var bools = List[Bool]()
    var i = 0
    while i < len(mask.vals):
        bools.append(mask.vals[i])
        i += 1
    return filter_rows(df, bools)


 

  

struct ColumnBoolOps:
    name: String
    fn __init__(out self, name: String):
        self.name = name
    fn ge(self, value: Int) -> Mask: return Mask()
    fn isin(self, values: List[String]) -> Mask: return Mask()
    fn between(self, lo: Int, hi: Int) -> Mask: return Mask()

 
 
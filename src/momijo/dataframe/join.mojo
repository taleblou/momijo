# Project:      Momijo
# Module:       dataframe.join
# File:         join.mojo
# Path:         dataframe/join.mojo
#
# Description:  dataframe.join — Relational join operations (inner, left, right, outer, semi/anti).
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
#   - Structs: ColMap, Acc
#   - Key functions: make_comp_key, make_bitmap, __init__, add_schema, append_row, materialize_two, next_f64, next_i64, next_bool, next_str, join_inner, join_left, join_right, join_anti, join_full, _build_map, _keys_of, _row_to_strings


from momijo.core.traits import one
from momijo.core.version import major
from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.column import Column, ColumnTag, F64, I64, STR, as_f64_or_nan, as_i64_or_zero, from_str, get_str
from momijo.dataframe.frame import DataFrame, get_column_at
from momijo.dataframe.helpers import m, t, unique
from momijo.dataframe.logical_plan import join
from momijo.dataframe.series_bool import SeriesBool
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr
from momijo.ir.dialects.annotations import tags
from momijo.nn.parameter import data
from momijo.tensor.broadcast import valid
from momijo.tensor.tensor import index
from momijo.utils.result import f, g

from collections.list import List


from momijo.dataframe.api import col_str, df_make,_find_col_idx
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import find_col


from momijo.dataframe.selection import RowRange, ColRange
from momijo.dataframe.helper import align_columns
from momijo.arrow_core.poly_column import get_string



struct Join:
    var left:  String
    var inner: String
    var right: String
    var outer: String
    var semi:  String
    var anti:  String

    fn __init__(out self):
        self.left  = String("left")
        self.inner = String("inner")
        self.right = String("right")
        self.outer = String("outer")
        self.semi  = String("semi")
        self.anti  = String("anti")


fn make_comp_key(df: DataFrame, cols: List[String], row: Int) -> String:
    # Build a composite key by concatenating key columns with a separator.
    var s = String("")
    var i = 0
    while i < len(cols):
        var c = df.get_column(cols[i])
        assert(c is not None, String("c is None"))
        s = s + get_string(c, row) + String("␟")
        i += 1
    return s

# Create a Bitmap from a boolean mask (True = valid, False = null).
fn make_bitmap(mask: List[Bool]) -> Bitmap:
    var bm = Bitmap(len(mask), True)
    var i = 0
    while i < len(mask):
        if not mask[i]:
            _ = bm.set(i, False)
        i += 1
    return bm

# Column mapping + accumulator

struct ColMap(Copyable, Movable):
    var tag: Int
    var idx: Int
fn __init__(out self):
        self.tag = ColumnTag.STR()
        self.idx = -1

# Acc holds schema (col_names/tags + per-type maps) and typed pools for values/valids.
struct Acc(Copyable, Movable):
    # schema
    var names_out: List[String]
    var tags_out:  List[Int]

# per-type column maps
    var m_f64: List[ColMap]
    var m_i64: List[ColMap]
    var m_b:   List[ColMap]
    var m_s:   List[ColMap]

# typed value pools (row-major append)
    var v_f64:  List[List[Float64]]
    var val_f64: List[List[Bool]]
    var v_i64:  List[List[Int64]]
    var val_i64: List[List[Bool]]
    var v_b:    List[List[Bool]]
    var val_b:   List[List[Bool]]
    var v_s:    List[List[String]]
    var val_s:   List[List[Bool]]
fn __init__(out self):
        self.names_out = List[String]()
        self.tags_out  = List[Int]()
        self.m_f64 = List[ColMap]()
        self.m_i64 = List[ColMap]()
        self.m_b   = List[ColMap]()
        self.m_s   = List[ColMap]()
        self.v_f64 = List[List[Float64]]()
        self.val_f64 = List[List[Bool]]()
        self.v_i64 = List[List[Int64]]()
        self.val_i64 = List[List[Bool]]()
        self.v_b   = List[List[Bool]]()
        self.val_b = List[List[Bool]]()
        self.v_s   = List[List[String]]()
        self.val_s = List[List[Bool]]()

# Append the schema of 'df' to this accumulator with a column-name suffix.
fn add_schema(mut self, df: DataFrame, suffix: String):
        var i = 0
        while i < df.ncols():
            var c = df.get_column_at(i)
            self.names_out.append(c.name() + suffix)
            self.tags_out.append(c.tag)

            var map = ColMap()
            map.tag = c.tag

            if c.tag == ColumnTag.F64():
                self.v_f64.append(List[Float64]())
                self.val_f64.append(List[Bool]())
                map.idx = len(self.v_f64) - 1
                self.m_f64.append(map)
            elif c.tag == ColumnTag.I64():
                self.v_i64.append(List[Int64]())
                self.val_i64.append(List[Bool]())
                map.idx = len(self.v_i64) - 1
                self.m_i64.append(map)
            elif c.tag == ColumnTag.BOOL():
                self.v_b.append(List[Bool]())
                self.val_b.append(List[Bool]())
                map.idx = len(self.v_b) - 1
                self.m_b.append(map)
            else:
                self.v_s.append(List[String]())
                self.val_s.append(List[Bool]())
                map.idx = len(self.v_s) - 1
                self.m_s.append(map)

            i += 1

# Append one row of data from 'df' into this accumulator.
# If 'valid_row' is False, we push nulls (valid=False) for all columns.
fn append_row(mut self, df: DataFrame, row: Int, valid_row: Bool):
        var i = 0
        var f = 0; var g = 0; var h = 0; var t = 0
        while i < df.ncols():
            var c = df.get_column_at(i)
            if c.tag == ColumnTag.F64():
                var idx = self.m_f64[f].idx
                if valid_row and c.is_valid(row):
                    self.v_f64[idx].append(c.as_f64_or_nan(row))
                    self.val_f64[idx].append(True)
                else:
                    self.v_f64[idx].append(0.0)
                    self.val_f64[idx].append(False)
                f += 1

            elif c.tag == ColumnTag.I64():
                var idxi = self.m_i64[g].idx
                if valid_row and c.is_valid(row):
                    self.v_i64[idxi].append(c.as_i64_or_zero(row))
                    self.val_i64[idxi].append(True)
                else:
                    self.v_i64[idxi].append(0)
                    self.val_i64[idxi].append(False)
                g += 1

            elif c.tag == ColumnTag.BOOL():
                var idxb = self.m_b[h].idx
                if valid_row and c.is_valid(row):
                    self.v_b[idxb].append(c.get_bool(row))
                    self.val_b[idxb].append(True)
                else:
                    self.v_b[idxb].append(False)
                    self.val_b[idxb].append(False)
                h += 1

            else:
                var idxs = self.m_s[t].idx
                if valid_row and c.is_valid(row):
                    self.v_s[idxs].append(c.get_str(row))
                    self.val_s[idxs].append(True)
                else:
                    self.v_s[idxs].append(String(""))
                    self.val_s[idxs].append(False)
                t += 1

            i += 1

# Build the final DataFrame by concatenating two accumulators (left then right).
fn materialize_two(accL: Acc, accR: Acc) -> DataFrame:
    var names_out = List[String]()
    var tags_out  = List[Int]()

# concat col_names/tags
    var i = 0
    while i < len(accL.names_out): names_out.append(accL.names_out[i]); i += 1
    i = 0
    while i < len(accR.names_out): names_out.append(accR.names_out[i]); i += 1

    i = 0
    while i < len(accL.tags_out): tags_out.append(accL.tags_out[i]); i += 1
    i = 0
    while i < len(accR.tags_out): tags_out.append(accR.tags_out[i]); i += 1

# build columns in order using tags_out
    var cols = List[Column]()
    var fiL = 0; var fiR = 0
    var iiL = 0; var iiR = 0
    var biL = 0; var biR = 0
    var siL = 0; var siR = 0

# helpers to pop from the correct side (L until exhausted for that type, then R)
fn next_f64() -> (List[Float64], List[Bool], Bool):
        if fiL < len(accL.v_f64):
            var v = accL.v_f64[fiL]; var m = accL.val_f64[fiL]; fiL += 1; return (v, m, True)
        else:
            var v2 = accR.v_f64[fiR]; var m2 = accR.val_f64[fiR]; fiR += 1; return (v2, m2, False)
fn next_i64() -> (List[Int64], List[Bool], Bool):
        if iiL < len(accL.v_i64):
            var v = accL.v_i64[iiL]; var m = accL.val_i64[iiL]; iiL += 1; return (v, m, True)
        else:
            var v2 = accR.v_i64[iiR]; var m2 = accR.val_i64[iiR]; iiR += 1; return (v2, m2, False)
fn next_bool() -> (List[Bool], List[Bool], Bool):
        if biL < len(accL.v_b):
            var v = accL.v_b[biL]; var m = accL.val_b[biL]; biL += 1; return (v, m, True)
        else:
            var v2 = accR.v_b[biR]; var m2 = accR.val_b[biR]; biR += 1; return (v2, m2, False)
fn next_str() -> (List[String], List[Bool], Bool):
        if siL < len(accL.v_s):
            var v = accL.v_s[siL]; var m = accL.val_s[siL]; siL += 1; return (v, m, True)
        else:
            var v2 = accR.v_s[siR]; var m2 = accR.val_s[siR]; siR += 1; return (v2, m2, False)

    var k = 0
    while k < len(names_out):
        var nm = names_out[k]
        var tg = tags_out[k]

        if tg == ColumnTag.F64():
            var tup = next_f64()
            var bm = make_bitmap(tup[1])
            var s = SeriesF64(nm, tup[0], bm)
            var c = Column(); c.from_f64(s)
            cols.append(c)

        elif tg == ColumnTag.I64():
            var tup = next_i64()
            var bm = make_bitmap(tup[1])
            var s = SeriesI64(nm, tup[0], bm)
            var c = Column(); c.from_i64(s)
            cols.append(c)

        elif tg == ColumnTag.BOOL():
            var tup = next_bool()
            var bm = make_bitmap(tup[1])
            var s = SeriesBool(nm, tup[0], bm)
            var c = Column(); c.from_bool(s)
            cols.append(c)

        else:
            var tup = next_str()
            var bm = make_bitmap(tup[1])
            var s = SeriesStr(nm, tup[0], bm)
            var c = Column(); c.from_str(s)
            cols.append(c)

        k += 1

    return DataFrame(names_out, cols)

# Join implementations
fn join_inner(left: DataFrame, right: DataFrame,
              left_keys: List[String], right_keys: List[String],
              suffix_left: String, suffix_right: String) -> DataFrame:
    var nL = left.nrows()
    var nR = right.nrows()

# Build right key index: unique key -> list of right-row indices
    var rk = List[String]()
    var i = 0
    while i < nR:
        rk.append(make_comp_key(right, right_keys, i))
        i += 1
    var uniq = List[String]()
    var groups = List[List[Int]]()
    i = 0
    while i < nR:
        var key = rk[i]
        var gid = -1
        var j = 0
        while j < len(uniq):
            if uniq[j] == key: gid = j; break
            j += 1
        if gid < 0:
            uniq.append(key)
            groups.append(List[Int]())
            gid = len(uniq) - 1
        groups[gid].append(i)
        i += 1

# Prepare accumulators
    var accL = Acc(); accL.add_schema(left,  suffix_left)
    var accR = Acc(); accR.add_schema(right, suffix_right)

# Probe left rows
    i = 0
    while i < nL:
        var lk = make_comp_key(left, left_keys, i)
        var gid = -1
        var u = 0
        while u < len(uniq):
            if uniq[u] == lk: gid = u; break
            u += 1
        if gid >= 0:
            var arr = groups[gid]
            var m = 0
            while m < len(arr):
                var rr = arr[m]
                accL.append_row(left,  i,  True)
                accR.append_row(right, rr, True)
                m += 1
        i += 1

    return materialize_two(accL, accR)
fn join_left(left: DataFrame, right: DataFrame,
             left_keys: List[String], right_keys: List[String],
             suffix_left: String, suffix_right: String) -> DataFrame:
    var nL = left.nrows()
    var nR = right.nrows()

    var rk = List[String]()
    var i = 0
    while i < nR:
        rk.append(make_comp_key(right, right_keys, i))
        i += 1
    var uniq = List[String]()
    var groups = List[List[Int]]()
    i = 0
    while i < nR:
        var key = rk[i]
        var gid = -1
        var j = 0
        while j < len(uniq):
            if uniq[j] == key: gid = j; break
            j += 1
        if gid < 0:
            uniq.append(key)
            groups.append(List[Int]())
            gid = len(uniq) - 1
        groups[gid].append(i)
        i += 1

    var accL = Acc(); accL.add_schema(left,  suffix_left)
    var accR = Acc(); accR.add_schema(right, suffix_right)

    i = 0
    while i < nL:
        var lk = make_comp_key(left, left_keys, i)
        var gid = -1
        var u = 0
        while u < len(uniq):
            if uniq[u] == lk: gid = u; break
            u += 1
        if gid >= 0:
            var arr = groups[gid]
            var m = 0
            while m < len(arr):
                var rr = arr[m]
                accL.append_row(left,  i,  True)
                accR.append_row(right, rr, True)
                m += 1
        else:
            accL.append_row(left,  i, True)
            accR.append_row(right, 0, False)  # pad right with nulls
        i += 1

    return materialize_two(accL, accR)

# Right join via left join with swapped sides/suffixes/keys.
fn join_right(left: DataFrame, right: DataFrame,
              left_keys: List[String], right_keys: List[String],
              suffix_left: String, suffix_right: String) -> DataFrame:
    return join_left(right, left, right_keys, left_keys, suffix_right, suffix_left)

# Anti join: left rows with NO match in right. Only left columns returned (no suffix).
fn join_anti(left: DataFrame, right: DataFrame,
             left_keys: List[String], right_keys: List[String]) -> DataFrame:
    var nL = left.nrows()
    var nR = right.nrows()

    var rk = List[String]()
    var i = 0
    while i < nR:
        rk.append(make_comp_key(right, right_keys, i))
        i += 1
    var uniq = List[String]()
    i = 0
    while i < nR:
        var key = rk[i]
        var found = False
        var j = 0
        while j < len(uniq):
            if uniq[j] == key: found = True; break
            j += 1
        if not found: uniq.append(key)
        i += 1

    var acc = Acc(); acc.add_schema(left, String(""))

    i = 0
    while i < nL:
        var lk = make_comp_key(left, left_keys, i)
        var exists = False
        var u = 0
        while u < len(uniq):
            if uniq[u] == lk: exists = True; break
            u += 1
        if not exists:
            acc.append_row(left, i, True)
        i += 1

# materialize only 'acc' (no right)
    var empty = Acc()
    return materialize_two(acc, empty)
fn join_full(left: DataFrame, right: DataFrame,
             left_keys: List[String], right_keys: List[String],
             suffix_left: String, suffix_right: String) -> DataFrame:
    var nL = left.nrows()
    var nR = right.nrows()

    var rk = List[String]()
    var i = 0
    while i < nR:
        rk.append(make_comp_key(right, right_keys, i))
        i += 1
    var uniq = List[String]()
    var groups = List[List[Int]]()
    i = 0
    while i < nR:
        var key = rk[i]
        var gid = -1
        var j = 0
        while j < len(uniq):
            if uniq[j] == key: gid = j; break
            j += 1
        if gid < 0:
            uniq.append(key)
            groups.append(List[Int]())
            gid = len(uniq) - 1
        groups[gid].append(i)
        i += 1

    var accL = Acc(); accL.add_schema(left,  suffix_left)
    var accR = Acc(); accR.add_schema(right, suffix_right)

    var matchedR = List[Bool]()
    i = 0
    while i < nR:
        matchedR.append(False)
        i += 1

    i = 0
    while i < nL:
        var lk = make_comp_key(left, left_keys, i)
        var gid = -1
        var u = 0
        while u < len(uniq):
            if uniq[u] == lk: gid = u; break
            u += 1

        if gid >= 0:
            var arr = groups[gid]
            var m = 0
            while m < len(arr):
                var rr = arr[m]
                accL.append_row(left,  i,  True)
                accR.append_row(right, rr, True)
                matchedR[rr] = True
                m += 1
        else:
            accL.append_row(left,  i, True)
            accR.append_row(right, 0, False)
        i += 1

# Append unmatched right rows
    var rr = 0
    while rr < nR:
        if not matchedR[rr]:
            accL.append_row(left,  0, False)
            accR.append_row(right, rr, True)
        rr += 1

    return materialize_two(accL, accR)



# ------------------------------ Internals -----------------------------------

# Build positions-per-unique-key (grouped row indices).
# Returns a list-of-lists of row indices; the implicit key order is the same
# as first appearance in the data (stable).
fn _build_map(df: DataFrame, key: String) -> List[List[Int]]:
   varidx = find_col(df, key)
    var keys = List[String]()
    var poslist = List[List[Int]]()
    var r: Int = 0
    while r < df.nrows():
       vark = df.cols[idx][r]
        var pos: Int = -1
        var i: Int = 0
        while i < len(keys):
            if keys[i] == k:
                pos = i
                break
            i += 1
        if pos < 0:
            keys.append(k)
            poslist.append(List[Int]())
            pos = len(keys) - 1
        poslist[pos].append(r)
        r += 1
    return poslist  # index paired with implicit 'keys' order

# Extract unique keys in first-seen order.
fn _keys_of(df: DataFrame, key: String) -> List[String]:
   varidx = find_col(df, key)
    var seen = List[String]()
    var r: Int = 0
    while r < df.nrows():
       vark = df.cols[idx][r]
        var dup: Bool = False
        var i: Int = 0
        while i < len(seen):
            if seen[i] == k:
                dup = True
                break
            i += 1
        if not dup:
            seen.append(k)
        r += 1
    return seen

# Materialize a DataFrame row to strings in column order.
fn _row_to_strings(df: DataFrame, r: Int) -> List[String]:
    var vals = List[String]()
    var c: Int = 0
    while c < df.ncols():
        vals.append(df.cols[c][r])
        c += 1
    return vals

# Append a materialized row (strings) into the output columns.
fn _append_row(cols: List[Column], row: List[String]):
    var i: Int = 0
    while i < len(cols):
        cols[i].append_str(row[i])
        i += 1

# Emit an empty row (all empty strings) of length n.
fn _empty_row(n: Int) -> List[String]:
    var vals = List[String]()
    var i: Int = 0
    while i < n:
        vals.append(String(""))
        i += 1
    return vals

# Build output schema: col_names = a.col_names + (b.col_names without key_b)
# and create empty string columns with those col_names.
fn _build_output_schema(a: DataFrame, b: DataFrame, key_b: String) -> (List[String], List[Column]):
    var col_names = List[String]()
    var cols = List[Column]()
    var c: Int = 0
    while c < a.ncols():
        col_names.append(a.col_names[c])
        cols.append(col_str(a.col_names[c], List[String]()))
        c += 1
   varib = find_col(b, key_b)
    var cb: Int = 0
    while cb < b.ncols():
        if cb != ib:
            col_names.append(b.col_names[cb])
            cols.append(col_str(b.col_names[cb], List[String]()))
        cb += 1
    return (col_names, cols)

# ------------------------------- Joins --------------------------------------

# INNER JOIN (single key)
fn inner_join(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame:
   varia = find_col(a, key_a)
   varib = find_col(b, key_b)
   varschema = _build_output_schema(a, b, key_b)
    var col_names = schema[0]
    var cols = schema[1]

    var rb: Int = 0
    while rb < b.nrows():
       vark = b.cols[ib][rb]
        var ra: Int = 0
        while ra < a.nrows():
            if a.cols[ia][ra] == k:
                var row = _row_to_strings(a, ra)
                var c2: Int = 0
                while c2 < b.ncols():
                    if c2 != ib:
                        row.append(b.cols[c2][rb])
                    c2 += 1
                _append_row(cols, row)
            ra += 1
        rb += 1
    return df_make(col_names, cols)

# ------------------------ Utilities ------------------------


# Build a collision-free row key by length-prefixing each cell
fn _row_key(df: DataFrame, r: Int) -> String:
    var key = String("")
    var c: Int = 0
    while c < df.ncols():
        var v = df.cols[c][r]
        key = key + String(len(v)) + String("#") + v
        c += 1
    return key

# Same as above but for a provided row buffer
fn _row_key_from_list(row: List[String]) -> String:
    var key = String("")
    var i: Int = 0
    while i < len(row):
        var v = row[i]
        key = key + String(len(v)) + String("#") + v
        i += 1
    return key

# Append one complete string-row (List[String]) into Column list
fn _append_row(mut cols: List[Column], row: List[String]) -> None:
    var c: Int = 0
    while c < len(cols):
        cols[c].push_string(row[c])
        c += 1

# Convert df row r to a List[String]
fn _row_to_strings(df: DataFrame, r: Int) -> List[String]:
    var out = List[String]()
    var c: Int = 0
    while c < df.ncols():
        out.append(df.cols[c][r])
        c += 1
    return out

# Construct empty strings row of given length
fn _empty_row(width: Int) -> List[String]:
    var out = List[String]()
    var i: Int = 0
    while i < width:
        out.append(String(""))
        i += 1
    return out

# Create an output schema: column names + empty string columns
# Drops the right key (key_b) to avoid duplicate key columns.
fn _build_output_schema(a: DataFrame, b: DataFrame, key_b: String) -> (List[String], List[Column]):
    var names = List[String]()
    var cols  = List[Column]()

    # all left names
    var i: Int = 0
    while i < a.ncols():
        names.append(a.col_names[i])
        cols.append(col_str(a.col_names[i], List[String]()))
        i += 1

    # all right names except key_b
    var j: Int = 0
    while j < b.ncols():
        if b.col_names[j] != key_b:
            names.append(b.col_names[j])
            cols.append(col_str(b.col_names[j], List[String]()))
        j += 1

    return (names, cols)



# ------------------------ LEFT JOIN ------------------------
# Keep all rows from 'a'; if no match in 'b', fill right fields with empty strings.
fn left_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame:
    var ia = find_col(a, key_a)
    var ib = find_col(b, key_b)
    if ia < 0 or ib < 0:
        # If key missing, degenerate to left with empties.
        var schema = _build_output_schema(a, b, key_b)
        var col_names = schema[0]
        var cols = schema[1]

        var ra: Int = 0
        while ra < a.nrows():
            var row2 = _row_to_strings(a, ra)
            var gaps = _empty_row(b.ncols() - 1)
            var i: Int = 0
            while i < len(gaps):
                row2.append(gaps[i])
                i += 1
            _append_row(cols, row2)
            ra += 1
        return df_make(col_names, cols)

    var schema2 = _build_output_schema(a, b, key_b)
    var col_names2 = schema2[0]
    var cols2 = schema2[1]

    var ra: Int = 0
    while ra < a.nrows():
        var ka = a.cols[ia][ra]
        var matched: Bool = False

        var rb: Int = 0
        while rb < b.nrows():
            if b.cols[ib][rb] == ka:
                matched = True
                var row = _row_to_strings(a, ra)
                var c2: Int = 0
                while c2 < b.ncols():
                    if c2 != ib:
                        row.append(b.cols[c2][rb])
                    c2 += 1
                _append_row(cols2, row)
            rb += 1

        if not matched:
            var row2 = _row_to_strings(a, ra)
            var gaps = _empty_row(b.ncols() - 1)
            var i2: Int = 0
            while i2 < len(gaps):
                row2.append(gaps[i2])
                i2 += 1
            _append_row(cols2, row2)

        ra += 1

    return df_make(col_names2, cols2)


# ------------------------ RIGHT via LEFT (swap) ------------------------
fn right_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame:
    return left_join_full(b, a, key_b, key_a)


# ------------------------ OUTER (FULL) JOIN ------------------------
# Union of left-join plus "right-only" rows (those without a match in 'a').
fn outer_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame:
    var left = left_join_full(a, b, key_a, key_b)

    # Build "right-only" by finding rows of b whose key has no match in a.
    var ia = find_col(a, key_a)
    var ib = find_col(b, key_b)

    var right_only: DataFrame
    do:
        var schema = _build_output_schema(a, b, key_b)
        var col_names = schema[0]
        var cols = schema[1]

        if ia < 0 or ib < 0:
            # If keys missing, treat everything as left and nothing on right-only.
            right_only = df_make(col_names, cols)
            break

        var r: Int = 0
        while r < b.nrows():
            var kb = b.cols[ib][r]

            # scan A for this key
            var found: Bool = False
            var ra: Int = 0
            while ra < a.nrows():
                if a.cols[ia][ra] == kb:
                    found = True
                    break
                ra += 1

            if not found:
                # materialize one "left-empties + right row (except key)" row
                var row = List[String]()
                # left empties
                var i: Int = 0
                while i < a.ncols():
                    row.append(String(""))
                    i += 1
                # right (skip key_b)
                var j: Int = 0
                while j < b.ncols():
                    if j != ib:
                        row.append(b.cols[j][r])
                    j += 1
                _append_row(cols, row)
            r += 1

        right_only = df_make(col_names, cols)
    end

    # Naive union with duplicate prevention using row-string keys (length-prefixed)
    var out_names = left.col_names
    var out_cols = List[Column]()
    var c: Int = 0
    while c < left.ncols():
        out_cols.append(col_str(out_names[c], List[String]()))
        c += 1

    var seen = List[String]()

    fn append_df_unique(df: DataFrame):
        var r: Int = 0
        while r < df.nrows():
            var key_str = _row_key(df, r)
            var exists: Bool = False
            var i: Int = 0
            while i < len(seen):
                if seen[i] == key_str:
                    exists = True
                    break
                i += 1
            if not exists:
                var row = _row_to_strings(df, r)
                _append_row(out_cols, row)
                seen.append(key_str)
            r += 1

    append_df_unique(left)
    append_df_unique(right_only)

    return df_make(out_names, out_cols)

# As-of join (left join with nearest match on key)
fn asof_join(left: DataFrame, right: DataFrame, on: String, by: String = String("")) -> DataFrame:
    """Perform an as-of join between left and right DataFrames.

    For each row in left, find the last row in right such that:
        right[on] <= left[on]
    If `by` is provided, matches are restricted to rows with same `by` value.
    """
    var out = DataFrame()
    out.col_names = List[String]()
    out.cols = List[List[String]]()

# Copy left columns first
    var c = 0
    while c < left.ncols():
        out.col_names.append(String(left.col_names[c]))
        var new_col = List[String]()
        var r = 0
        while r < left.nrows():
            new_col.append(String(left.cols[c][r]))
            r += 1
        out.cols.append(new_col)
        c += 1

# Prepare to add right-only columns
    var right_only = List[Int]()
    var rc = 0
    while rc < right.ncols():
        var name = right.col_names[rc]
        if name != on and (by == String("") or name != by):
            out.col_names.append(name)
            right_only.append(rc)
        rc += 1

# Initialize new right-only cols with blanks
    var k = 0
    while k < len(right_only):
        var blank = List[String]()
        var r2 = 0
        while r2 < left.nrows():
            blank.append(String(""))
            r2 += 1
        out.cols.append(blank)
        k += 1

# Perform as-of matching
    var i = 0
    while i < left.nrows():
        var left_key = left.cols[_find_col(left, on)][i]
        var group_val = String("")
        if by != String(""):
            group_val = left.cols[_find_col(left, by)][i]

# Find candidate in right
        var best_j = -1
        var j = 0
        while j < right.nrows():
            var rk = right.cols[_find_col(right, on)][j]
            if rk <= left_key:
                if by == String("") or right.cols[_find_col(right, by)][j] == group_val:
                    best_j = j
            j += 1

# Fill matched values if found
        if best_j >= 0:
            var ri = 0
            while ri < len(right_only):
                var col_idx = right_only[ri]
                var val = right.cols[col_idx][best_j]
                out.cols[left.ncols() + ri][i] = String(val)
                ri += 1
        i += 1

    return out


# Internal helper: find column index
fn _find_col(df: DataFrame, name: String) -> Int:
    var i = 0
    while i < len(df.col_names):
        if df.col_names[i] == name:
            return i
        i += 1
    return -1
fn make_comp_key(df: DataFrame, cols: List[String], row: Int) -> String:
    var s_out = String("")
    var i: Int = 0
    while i < len(cols):
        var c = df.get_column(cols[i])      # Column
        var v = get_string(c, row)
        s_out = s_out + String(len(v)) + String("#") + v
        i += 1
    return s_out


fn make_bitmap(mask: List[Bool]):
    var bm = Bitmap(len(mask), True)
    var i = 0
    while i < len(mask):
        if not mask[i]:
            _ = bm.set(i, False)
        i += 1
    return bm

fn _build_map(df: DataFrame, key: String):
    var idx = _find_col(df, key)
    var keys = List[String]()
    var poslist = List[List[Int]]()
    var r = 0
    while r < df.nrows():
        var k = df.cols[idx][r]
        var pos = -1
        var i = 0
        while i < len(keys):
            if keys[i] == k:
                pos = i
                break
            i += 1
        if pos < 0:
            keys.append(k)
            var v = List[Int]()
            v.append(r)
            poslist.append(v)
        else:
            poslist[pos].append(r)
        r += 1
    return poslist

fn _keys_of(df: DataFrame, key: String):
    var idx = _find_col(df, key)
    var out = List[String]()
    var r = 0
    while r < df.nrows():
        out.append(df.cols[idx][r])
        r += 1
    return out

fn _row_to_strings(df: DataFrame, r: Int):
    var row = List[String]()
    var c = 0
    while c < df.ncols():
        row.append(df.cols[c][r])
        c += 1
    return row

fn _empty_row(n: Int):
    var row = List[String]()
    var i = 0
    while i < n:
        row.append(String(""))
        i += 1
    return row

fn inner_join(a: DataFrame, b: DataFrame, key_a: String, key_b: String):
    var ia = _find_col(a, key_a)
    var ib = _find_col(b, key_b)
    var schema = _build_output_schema(a, b, key_b)
    var col_names = schema[0]
    var cols = schema[1]

    var rb = 0
    while rb < b.nrows():
        var kb = b.cols[ib][rb]
        var ra = 0
        while ra < a.nrows():
            if a.cols[ia][ra] == kb:
                var row = List[String]()
                var c = 0
                while c < a.ncols():
                    row.append(a.cols[c][ra])
                    c += 1
                var c2 = 0
                while c2 < b.ncols():
                    if c2 != ib:
                        row.append(b.cols[c2][rb])
                    c2 += 1
                _append_row(cols, row)
            ra += 1
        rb += 1
    return df_make(col_names, cols)

fn left_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String):
    var ia = _find_col(a, key_a)
    var ib = _find_col(b, key_b)
    var schema = _build_output_schema(a, b, key_b)
    var col_names = schema[0]
    var cols = schema[1]

    var empty_right = _empty_row(b.ncols() - 1)

    var ra = 0
    while ra < a.nrows():
        var ka = a.cols[ia][ra]
        var matched = False
        var rb = 0
        while rb < b.nrows():
            if b.cols[ib][rb] == ka:
                matched = True
                var row = List[String]()
                var c = 0
                while c < a.ncols():
                    row.append(a.cols[c][ra])
                    c += 1
                var c2 = 0
                while c2 < b.ncols():
                    if c2 != ib:
                        row.append(b.cols[c2][rb])
                    c2 += 1
                _append_row(cols, row)
            rb += 1
        if not matched:
            var row2 = List[String]()
            var c3 = 0
            while c3 < a.ncols():
                row2.append(a.cols[c3][ra])
                c3 += 1
            var j = 0
            while j < len(empty_right):
                row2.append(empty_right[j])
                j += 1
            _append_row(cols, row2)
        ra += 1

    return df_make(col_names, cols)

fn right_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String):
    var out = left_join_full(b, a, key_b, key_a)
    return out

fn outer_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame:
    var left_only  = left_join_full(a, b, key_a, key_b)
    var right_only = left_join_full(b, a, key_b, key_a)

    var col_names = left_only.col_names
    var cols = List[Column]()
    var c: Int = 0
    while c < left_only.ncols():
        cols.append(col_str(col_names[c], List[String]()))
        c += 1

    fn _append_row_copy(df: DataFrame, r: Int):
        var row = _row_to_strings(df, r)
        _append_row(cols, row)

    # seed output with left-only
    var r: Int = 0
    while r < left_only.nrows():
        _append_row_copy(left_only, r)
        r += 1

    # build seen set from left-only using length-prefixed keys
    var seen = List[String]()
    r = 0
    while r < left_only.nrows():
        var key_str = _row_key(left_only, r)
        seen.append(key_str)
        r += 1

    # append unique rows from right-only
    r = 0
    while r < right_only.nrows():
        var key_str2 = _row_key(right_only, r)
        var dup: Bool = False
        var i: Int = 0
        while i < len(seen):
            if seen[i] == key_str2:
                dup = True
                break
            i += 1
        if not dup:
            _append_row_copy(right_only, r)
            seen.append(key_str2)
        r += 1

    return df_make(col_names, cols)




fn merge(left: DataFrame, right: DataFrame, on: List[String], how: String) -> DataFrame:
    # Left-join only. 'how' is ignored for now.
    var keys = on.copy()

    # Map key columns to indices in both frames
    var lk = List[Int]()
    var rk = List[Int]()
    var i = 0
    while i < len(keys):
        lk.append(_find_col_idx(left,  keys[i]))
        rk.append(_find_col_idx(right, keys[i]))
        i += 1

    # ---- Build right index: parallel arrays (keys -> list of row ids) ----
    var rkeys  = List[String]()
    var rlists = List[List[Int]]()

    var rrow = 0
    while rrow < right.nrows():
        var key = String("")
        var j = 0
        while j < len(rk):
            var s = right.cols[rk[j]][rrow]
            if j > 0:
                key += String("\x1f")   # field separator
            key += s
            j += 1

        # Linear search (no Dict to avoid may-raise)
        var pos = -1
        var t = 0
        while t < len(rkeys):
            if rkeys[t] == key:
                pos = t
                break
            t += 1

        if pos < 0:
            var lst = List[Int]()
            lst.append(rrow)
            rkeys.append(key)
            rlists.append(lst.copy())
        else:
            rlists[pos].append(rrow)

        rrow += 1

    # ---- Output schema: left columns + right non-key columns ----
    var out_names = List[String]()
    i = 0
    while i < left.ncols():
        out_names.append(left.col_names[i])
        i += 1

    var right_nonkey = List[Int]()
    i = 0
    while i < right.ncols():
        var is_key = False
        var u = 0
        while u < len(rk):
            if i == rk[u]:
                is_key = True
                break
            u += 1
        if not is_key:
            right_nonkey.append(i)
            out_names.append(right.col_names[i])
        i += 1

    # Allocate output columns
    var out_vals = List[List[String]]()
    i = 0
    while i < len(out_names):
        out_vals.append(List[String]())
        i += 1

    # ---- Populate rows (left-join: take first match; else NA="") ----
    var r2 = 0
    while r2 < left.nrows():
        # Append left columns
        i = 0
        while i < left.ncols():
            out_vals[i].append(left.cols[i][r2])
            i += 1

        # Build left composite key
        var key2 = String("")
        var j3 = 0
        while j3 < len(lk):
            var s2 = left.cols[lk[j3]][r2]
            if j3 > 0:
                key2 += String("\x1f")
            key2 += s2
            j3 += 1

        # Find in right index
        var pos2 = -1
        var t2 = 0
        while t2 < len(rkeys):
            if rkeys[t2] == key2:
                pos2 = t2
                break
            t2 += 1

        # Append right non-key columns
        var offset = left.ncols()
        var rn = 0
        while rn < len(right_nonkey):
            var rc = right_nonkey[rn]
            if pos2 >= 0:
                var rr = rlists[pos2][0]                # first match
                out_vals[offset].append(right.cols[rc][rr])
            else:
                out_vals[offset].append(String(""))     # NA for left-join miss
            offset += 1
            rn += 1

        r2 += 1

    # Build DataFrame (reuse left index)
    return DataFrame(out_names, out_vals, left.index_vals, left.index_name)





struct AsOf:
    var backward: Int
    var forward: Int
    var nearest: Int
    fn __init__(out self):
        self.backward = 0
        self.forward = 1
        self.nearest = 2

fn _mid_lex(a: String, b: String) -> String:
    # var i = 0
    # var la = len(a)
    # var lb = len(b)
    # var lm = la
    # if lb < lm:
    #     lm = lb
    # while i < lm and a[i] == b[i]:
    #     i += 1
    # if i == lm:
    #     if la < lb:
    #         return b
    #     else:
    #         return a
    # var ch_a = a[i]
    # var ch_b = b[i]
    # var ch = ch_a
    # if ch_b > ch_a:
    #     ch = ch_b
    # var out = String("")
    # var j = 0
    # while j < i:
    #     out = out + String(a[j])
    #     j += 1
    # out = out + String(ch)
    return "out"

fn merge_asof(left: DataFrame, right: DataFrame, on: String, direction: Int) -> DataFrame:
    var lkey = left.find_col(on)
    var rkey = right.find_col(on)
    if lkey < 0 or rkey < 0:
        return left.copy()

    var nL = left.nrows()
    var nR = right.nrows()

    var r_keep_idx = List[Int]()
    var r_keep_names = List[String]()
    var c = 0
    while c < right.ncols():
        if c != rkey:
            r_keep_idx.append(c)
            r_keep_names.append(right.col_names[c])
        c += 1

    var r_keys = List[String]()
    r_keys.reserve(nR)
    var i = 0
    while i < nR:
        r_keys.append(right.cols[rkey].get_string(i))
        i += 1

    var out_names = List[String]()
    var out_vals_cols = List[List[String]]()

    var lc = 0
    while lc < left.ncols():
        out_names.append(left.col_names[lc])
        var buf = List[String]()
        buf.reserve(nL)
        out_vals_cols.append(buf.copy())
        lc += 1

    var rk = 0
    while rk < len(r_keep_idx):
        out_names.append(r_keep_names[rk])
        var buf2 = List[String]()
        buf2.reserve(nL)
        out_vals_cols.append(buf2.copy())
        rk += 1

    var rpos = 0
    var rpos_prev = -1
    var rpos_next = 0

    var lr = 0
    while lr < nL:
        var lk = left.cols[lkey].get_string(lr)
        var match_idx = -1

        if direction == AsOf().backward:
            while rpos < nR and r_keys[rpos] <= lk:
                rpos += 1
            match_idx = rpos - 1
        elif direction == AsOf().forward:
            while rpos < nR and r_keys[rpos] < lk:
                rpos += 1
            if rpos < nR:
                match_idx = rpos
            else:
                match_idx = -1
        else:
            while rpos_next < nR and r_keys[rpos_next] < lk:
                rpos_next += 1
            rpos_prev = rpos_next - 1

            var has_prev = (rpos_prev >= 0)
            var has_next = (rpos_next < nR)

            if has_prev and has_next:
                var pk = r_keys[rpos_prev]
                var nk = r_keys[rpos_next]
                if lk == pk:
                    match_idx = rpos_prev
                elif lk == nk:
                    match_idx = rpos_next
                else:
                    var left_span_smaller = (lk <= _mid_lex(pk, nk))
                    if left_span_smaller:
                        match_idx = rpos_prev
                    else:
                        match_idx = rpos_next
            elif has_prev:
                match_idx = rpos_prev
            elif has_next:
                match_idx = rpos_next
            else:
                match_idx = -1

        var lc2 = 0
        while lc2 < left.ncols():
            out_vals_cols[lc2].append(left.cols[lc2].get_string(lr))
            lc2 += 1

        var base = left.ncols()
        var rk2 = 0
        while rk2 < len(r_keep_idx):
            if match_idx >= 0:
                out_vals_cols[base + rk2].append(right.cols[r_keep_idx[rk2]].get_string(match_idx))
            else:
                out_vals_cols[base + rk2].append(String(""))
            rk2 += 1

        lr += 1

    var idx_vals = List[String]()
    return DataFrame(out_names.copy(), out_vals_cols.copy(), idx_vals.copy(), String(""))



# ------------------------ Merge (Left Join) ------------------------
# `on` contains column names present in both left and right.
# Only string equality joins are supported here.

fn left_join(left: DataFrame, right: DataFrame, on: List[String]) -> DataFrame:
# Build index for right on key
    var right_key_cols = List[List[String]]()
    var oi = 0
    while oi < len(on):
        right_key_cols.append(_safe_get_col(right, on[oi]))
        oi += 1

    var right_key_of_row = List[String]()
    var rr = 0
    while rr < right.nrows():
        var kv = List[String]()
        var kci = 0
        while kci < len(right_key_cols):
            kv.append(right_key_cols[kci][rr])
            kci += 1
        right_key_of_row.append(_make_key(kv))
        rr += 1

    var index = Dictionary[String, Int]()   # first matching row
    var seen = Set[String]()
    var r2 = 0
    while r2 < len(right_key_of_row):
        var k = right_key_of_row[r2]
        if not seen.contains(k):
            index[k] = r2
            seen.insert(k)
        r2 += 1

# Prepare output schema
    var out_names = List[String]()
    var out_cols = List[List[String]]()

# Start with all left columns
    var lc = 0
    while lc < left.ncols():
        out_names.append(left.col_names[lc])
        out_cols.append(List[String]())
        lc += 1

# Add right columns except the join keys (to avoid duplicates)
    var rc = 0
    while rc < right.ncols():
        var nm = right.col_names[rc]
        var is_key = False
        var kx = 0
        while kx < len(on):
            if on[kx] == nm:
                is_key = True
                break
            kx += 1
        if not is_key:
            out_names.append(nm)
            out_cols.append(List[String]())
        rc += 1

# Fill rows
    var left_key_cols = List[List[String]]()
    var oi2 = 0
    while oi2 < len(on):
        left_key_cols.append(_safe_get_col(left, on[oi2]))
        oi2 += 1

    var lr = 0
    while lr < left.nrows():
# Build left key
        var kv2 = List[String]()
        var kci2 = 0
        while kci2 < len(left_key_cols):
            kv2.append(left_key_cols[kci2][lr])
            kci2 += 1
        var lk = _make_key(kv2)

# Append all left values
        var lc2 = 0
        while lc2 < left.ncols():
            out_cols[lc2].append(left.cols[lc2][lr])
            lc2 += 1

# Append matching right values (or empty)
        var right_row = -1
        if index.contains(lk):
            right_row = index[lk]

        var out_pos = left.ncols()
        var rc2 = 0
        while rc2 < right.ncols():
            var nm2 = right.col_names[rc2]
            var is_key2 = False
            var kx2 = 0
            while kx2 < len(on):
                if on[kx2] == nm2:
                    is_key2 = True
                    break
                kx2 += 1
            if not is_key2:
                if right_row >= 0:
                    out_cols[out_pos].append(right.cols[rc2][right_row])
                else:
                    out_cols[out_pos].append(String(""))
                out_pos += 1
            rc2 += 1

        lr += 1

    return _df_from(out_names, out_cols)

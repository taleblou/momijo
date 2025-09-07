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
# File: src/momijo/dataframe/join.mojo

from momijo.core.error import module
from momijo.core.traits import one
from momijo.core.version import major
from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.column import Column, ColumnTag, F64, I64, STR, as_f64_or_nan, as_i64_or_zero, from_str, get_str, value_str
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
from pathlib import Path
from pathlib.path import Path
from sys import version

fn make_comp_key(df: DataFrame, cols: List[String], row: Int) -> String:
    # Build a composite key by concatenating key columns with a separator.
    var s = String("")
    var i = 0
    while i < len(cols):
        var c = df.get_column(cols[i])
        assert(c is not None, String("c is None"))
        s = s + c.value()_str(row) + String("␟")
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

# -----------------------------------------------------------------------------
# Column mapping + accumulator
# -----------------------------------------------------------------------------

struct ColMap(Copyable, Movable):
    var tag: Int
    var idx: Int
fn __init__(out self) -> None:
        self.tag = ColumnTag.STR()
        self.idx = -1

# Acc holds schema (names/tags + per-type maps) and typed pools for values/valids.
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
fn __init__(out self) -> None:
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
fn add_schema(mut self, df: DataFrame, suffix: String) -> None:
        var i = 0
        while i < df.width():
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
fn append_row(mut self, df: DataFrame, row: Int, valid_row: Bool) -> None:
        var i = 0
        var f = 0; var g = 0; var h = 0; var t = 0
        while i < df.width():
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

    # concat names/tags
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

# -----------------------------------------------------------------------------
# Join implementations
# -----------------------------------------------------------------------------
fn join_inner(left: DataFrame, right: DataFrame,
              left_keys: List[String], right_keys: List[String],
              suffix_left: String, suffix_right: String) -> DataFrame:
    var nL = left.height()
    var nR = right.height()

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
    var nL = left.height()
    var nR = right.height()

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
    var nL = left.height()
    var nR = right.height()

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
    var nL = left.height()
    var nR = right.height()

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
# Project:      Momijo
# Module:       dataframe.helpers
# File:         helpers.mojo
# Path:         dataframe/helpers.mojo
#
# Description:  dataframe.helpers — Helpers module for Momijo DataFrame.
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
#   - Structs: ModuleState
#   - Key functions: __dict_contains, __init__, make_module_state, contains_string, unique_strings_list, list_union, list_difference, symdiff, union, between_i64, isna, take_i64, take_f64, take_str, take_bool, argsort_f64, argsort_i64, rank_dense_f64

from collections.list import List
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.io_bytes import str_to_bytes, bytes_to_string
from collections.dict import Dict
from momijo.dataframe.selection import RowRange, ColRange   # kept-for-future
from momijo.dataframe._groupby_core import groupby_agg      # kept-for-future
from momijo.arrow_core.poly_column import get_string        # kept, project-wide
# helper: check key existence without raising
fn __dict_contains(m: Dict[String, List[Int]], key: String) -> Bool:
# Dict exposes keys(); we check by iterating
    for k in m.keys():
        if k == key:
            return True
    return False
# --------------------------- Module state (placeholder) ---------------------

struct ModuleState:
    var j: Int
    fn __init__(out self, j: Int = 0):
        self.j = j

fn make_module_state() -> ModuleState:
    return ModuleState(0)

 
# --------------------------- String list helpers ----------------------------

fn contains_string(xs: List[String], x: String) -> Bool:
    var i: Int = 0
    while i < len(xs):
        if xs[i] == x:
            return True
        i += 1
    return False

fn unique_strings_list(xs: List[String]) -> List[String]:
    var out = List[String]()
    var i: Int = 0
    while i < len(xs):
        if not contains_string(out, xs[i]):
            out.append(xs[i])
        i += 1
    return out

fn list_union(a: List[String], b: List[String]) -> List[String]:
    var out = unique_strings_list(a)
    var i: Int = 0
    while i < len(b):
        if not contains_string(out, b[i]):
            out.append(b[i])
        i += 1
    return out

fn list_difference(a: List[String], b: List[String]) -> List[String]:
    var out = List[String]()
    var i: Int = 0
    while i < len(a):
        if not contains_string(b, a[i]):
            out.append(a[i])
        i += 1
    return out

fn symdiff(a: List[String], b: List[String]) -> List[String]:
    var ab = list_difference(a, b)
    var ba = list_difference(b, a)
    return list_union(ab, ba)

fn union(a: List[String], b: List[String]) -> List[String]:
    return list_union(a, b)

# --------------------------- Masks & simple predicates ----------------------

fn between_i64(x: Int64, a: Int64, b: Int64) -> Bool:
    return x >= a and x <= b

fn isna(s: String) -> Bool:
# Simple NA heuristic used across project
    return (s == String("")) or (s == String("NA")) or (s == String("NaN"))

# --------------------------- Take (parallel index) --------------------------

fn take_i64(xs: List[Int64], idx: List[Int]) -> List[Int64]:
    var out = List[Int64]()
    var i: Int = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_f64(xs: List[Float64], idx: List[Int]) -> List[Float64]:
    var out = List[Float64]()
    var i: Int = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_str(xs: List[String], idx: List[Int]) -> List[String]:
    var out = List[String]()
    var i: Int = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_bool(xs: List[Bool], idx: List[Int]) -> List[Bool]:
    var out = List[Bool]()
    var i: Int = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

# --------------------------- Sorting & ranking ------------------------------

fn argsort_f64(xs: List[Float64], asc: Bool = True) -> List[Int]:
# Selection sort of indices for determinism and minimal deps
    var idxs = List[Int]()
    var i: Int = 0
    while i < len(xs):
        idxs.append(i)
        i += 1
    var a: Int = 0
    while a < len(xs):
        var b: Int = a + 1
        while b < len(xs):
            var cond = xs[idxs[b]] < xs[idxs[a]] if asc else xs[idxs[b]] > xs[idxs[a]]
            if cond:
                var t = idxs[a]; idxs[a] = idxs[b]; idxs[b] = t
            b += 1
        a += 1
    return idxs

fn argsort_i64(xs: List[Int64], asc: Bool = True) -> List[Int]:
    var idxs = List[Int]()
    var i: Int = 0
    while i < len(xs):
        idxs.append(i)
        i += 1
    var a: Int = 0
    while a < len(xs):
        var b: Int = a + 1
        while b < len(xs):
            var cond = xs[idxs[b]] < xs[idxs[a]] if asc else xs[idxs[b]] > xs[idxs[a]]
            if cond:
                var t = idxs[a]; idxs[a] = idxs[b]; idxs[b] = t
            b += 1
        a += 1
    return idxs
# insertion-sort indices for Int64
fn argsort_i64(xs: List[Int64]) -> List[Int]
    var idx = List[Int]()
    var i = 0
    while i < len(xs):
        idx.append(i)
        i += 1
    var j = 1
    while j < len(idx):
        var key = idx[j]
        var k = j - 1
        while k >= 0 and xs[idx[k]] > xs[key]:
            idx[k + 1] = idx[k]
            k -= 1
        idx[k + 1] = key
        j += 1
    return idx

    
fn rank_dense_f64(xs: List[Float64]) -> List[Int]:
    var order = argsort_f64(xs, True)
    var ranks = List[Int]()
# init ranks with zeros
    var i: Int = 0
    while i < len(xs):
        ranks.append(0)
        i += 1
    var rnk: Int = 0
    var k: Int = 0
    while k < len(order):
        if k == 0 or xs[order[k]] != xs[order[k - 1]]:
            rnk += 1
        ranks[order[k]] = rnk
        k += 1
    return ranks

# --------------------------- Math & stats -----------------------------------

fn corr(x: List[Float64], y: List[Float64]) -> Float64:
   varn = len(x)
    if n == 0 or len(y) != n:
        return 0.0
    var sx: Float64 = 0.0
    var sy: Float64 = 0.0
    var i: Int = 0
    while i < n:
        sx += x[i]; sy += y[i]
        i += 1
   varmx = sx / Float64(n)
   varmy = sy / Float64(n)
    var num: Float64 = 0.0
    var vx: Float64 = 0.0
    var vy: Float64 = 0.0
    i = 0
    while i < n:
       vardx = x[i] - mx
       vardy = y[i] - my
        num += dx * dy
        vx += dx * dx
        vy += dy * dy
        i += 1
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return num / (vx.sqrt() * vy.sqrt())

# --------------------------- Search -----------------------------------------

fn searchsorted_f64(sorted: List[Float64], x: Float64) -> Int:
    var i: Int = 0
    while i < len(sorted) and sorted[i] < x:
        i += 1
    return i

# --------------------------- DataFrame utilities ----------------------------

fn find_col(df: DataFrame, name: String) -> Int:
    var i: Int = 0
    while i < df.ncols():
        if df.col_names[i] == name:
            return i
        i += 1
    return -1

fn df_cell(df: DataFrame, c: Int, r: Int) -> String:
# Safe textual accessor
    return df.cols[c][r]

fn header(t: String):
    print(String("\n==================== ") + t + String(" ===================="))

# --------------------------- Byte/encoding re-exports -----------------------

fn str_to(s: String) -> List[UInt8]:
    return str_to_bytes(s)

fn u32_to_le(x: UInt32) -> List[UInt8]:
    return u32_to_le_bytes(x)

fn u64_to_le(x: UInt64) -> List[UInt8]:
    return u64_to_le_bytes(x)

fn i64_to_le(x: Int64) -> List[UInt8]:
    return i64_to_le_bytes(x)


fn rep(s: String, n: Int) -> String:
    var out = String("")
    var i: Int = 0
    while i < n:
        out = out + s
        i += 1
    return out

fn pad(s: String, w: Int) -> String:
    var out = s
    var i: Int = len(out)
    while i < w:
        out = out + String(" ")
        i += 1
    return out

fn join_with(xs: List[String], sep: String) -> String:
    var out = String("")
    var i: Int = 0
    while i < len(xs):
        out = out + xs[i]
        if i + 1 < len(xs):
            out = out + sep
        i += 1
    return out

# -------------------------- Column/DF diagnostics ---------------------------

fn dtype_name_of_column(c: Column) -> String:
# Prefer tag-based inspection so it stays cheap and portable.
   vart = c.tag()
    if t == ColumnTag.I64():
        return String("Int64")
    if t == ColumnTag.F64():
        return String("Float64")
    if t == ColumnTag.BOOL():
        return String("Bool")
    if t == ColumnTag.STR():
        return String("String")
    return String("<unknown>")

fn join_col_names(df: DataFrame) -> String:
# If DataFrame exposes .col_names (List[String]) use it; otherwise call name accessors.
    var col_names = List[String]()
    var c: Int = 0
   varC = ncols(df)
    while c < C:
# df_cell returns cell strings; for col_names we assume DataFrame has .col_names
# To keep broad compatibility, append from df.col_names if available.
# Here we assume df.col_names exists and is List[String].
        col_names.append(df.col_names[c])
        c += 1
    return join_with(col_names, String(", "))

fn print_df_info(df_name: String, df: DataFrame):
    print(String("== ") + df_name + String(" =="))
    print(String("shape: (") + String(nrows(df)) + String(", ") + String(ncols(df)) + String(")"))
    print(String("columns: ") + join_col_names(df))
# Per-column dtype
    var i: Int = 0
    while i < ncols(df):
        print(df.col_names[i] + String(": ") + dtype_name_of_column(df.cols[i]))
        i += 1

# ------------------------------ List 'head' ---------------------------------

fn print_i64_head(label: String, xs: List[Int64], k: Int):
    var n = len(xs)
    var lim = k
    if lim > n:
        lim = n

    var line = label + String(": [")
    var i: Int = 0
    while i < lim:
        line = line + String(xs[i])
        if i + 1 < lim:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)

fn print_f64_head(label: String, xs: List[Float64], k: Int):
    var n = len(xs)
    var lim = k
    if lim > n:
        lim = n

    var line = label + String(": [")
    var i: Int = 0
    while i < lim:
        line = line + String(xs[i])
        if i + 1 < lim:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)

fn print_str_head(label: String, xs: List[String], k: Int):
    var n = len(xs)
    var lim = k
    if lim > n:
        lim = n

    var line = label + String(": [")
    var i: Int = 0
    while i < lim:
        line = line + xs[i]
        if i + 1 < lim:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)

# ------------------------------ List joiners --------------------------------

fn join_i64_list(xs: List[Int64]) -> String:
    var out = String("[")
    var i: Int = 0
    while i < len(xs):
        out = out + String(xs[i])
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn join_f64_list(xs: List[Float64]) -> String:
    var out = String("[")
    var i: Int = 0
    while i < len(xs):
        out = out + String(xs[i])
        if i + 1 < len(xs):
            out = out + String(", ")
        i += 1
    out = out + String("]")
    return out

fn join_bool_list(xs: List[Bool]) -> String:
    var out = String("[")
    var i: Int = 0
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

# ----------------------------- Pretty printing ------------------------------

fn header(t: String):
    print(String("\n") + String("==================== ") + t + String(" ===================="))

fn safe(op: String, ok: Bool):
    if ok:
        print(String("- ") + op + String(" ... OK"))
    else:
        print(String("- ") + op + String(" ... FAIL"))

# ASCII table printer for DataFrame.
# mode: "head" (default), "tail", "all"
fn print_df(df: DataFrame, rows: Int, mode: String = String("head")):
# column widths
   varcol_w: Int = 14
   varleft_w: Int = 6

   varC = ncols(df)
   varR = nrows(df)

    var start_row: Int = 0
    var count: Int = rows

    if mode == String("all"):
        count = R
        start_row = 0
    else:
        if count > R:
            count = R
        if mode == String("tail"):
            start_row = R - count
            if start_row < 0:
                start_row = 0

    print(String("\n=== DataFrame ==="))
    print(String("shape: (") + String(R) + String(", ") + String(C) + String(")"))
    print(String("columns: ") + join_col_names(df))

# Borders
    var top = String("+") + rep(String("-"), left_w)
    var mid = String("+") + rep(String("="), left_w)
    var i: Int = 0
    while i < C:
        top = top + String("+") + rep(String("-"), col_w)
        mid = mid + String("+") + rep(String("="), col_w)
        i += 1
    top = top + String("+")
    mid = mid + String("+")

    print(top)

# Header row
    var hdr = String("|") + pad(String("#"), left_w)
    i = 0
    while i < C:
        hdr = hdr + String("|") + pad(df.col_names[i], col_w)
        i += 1
    hdr = hdr + String("|")
    print(hdr)
    print(mid)

# Rows
    var r: Int = 0
    while r < count:
        var line = String("|") + pad(String("#") + String(start_row + r), left_w)
        var cc: Int = 0
        while cc < C:
            line = line + String("|") + pad(df_cell(df, cc, start_row + r), col_w)
            cc += 1
        line = line + String("|")
        print(line)
        r += 1

    print(top)

# Align column names
fn align_columns(left: DataFrame, right: DataFrame) -> (DataFrame, DataFrame):
    """Ensure both frames have the same column set and order.
    Missing columns in either frame are created and filled with empty strings.
    The column order in both outputs follows the union order: first left's columns,
    then any additional columns from right in their original order.
    """
    var left_copy = left.copy()
    var right_copy = right.copy()

# Collect union of all column names (left order first, then right extras)
    var all_cols = List[String]()
    var i = 0
    while i < len(left.col_names):
        var name = left.col_names[i]
        if all_cols.count(name) == 0:
            all_cols.append(name)
        i += 1
    var j = 0
    while j < len(right.col_names):
        var name2 = right.col_names[j]
        if all_cols.count(name2) == 0:
            all_cols.append(name2)
        j += 1

# Ensure left_copy has all columns
    var i2 = 0
    while i2 < len(all_cols):
        var col = all_cols[i2]
# find index of col in left_copy
        var idx = -1
        var k = 0
        while k < len(left_copy.col_names):
            if left_copy.col_names[k] == col:
                idx = k
                break
            k += 1
        if idx == -1:
            left_copy.col_names.append(col)
            var filler = List[String]()
            var r = 0
            while r < left_copy.nrows():
                filler.append(String(""))
                r += 1
            left_copy.cols.append(filler)
        i2 += 1

# Ensure right_copy has all columns
    var j2 = 0
    while j2 < len(all_cols):
        var col2 = all_cols[j2]
        var idx2 = -1
        var k2 = 0
        while k2 < len(right_copy.col_names):
            if right_copy.col_names[k2] == col2:
                idx2 = k2
                break
            k2 += 1
        if idx2 == -1:
            right_copy.col_names.append(col2)
            var filler2 = List[String]()
            var r2 = 0
            while r2 < right_copy.nrows():
                filler2.append(String(""))
                r2 += 1
            right_copy.cols.append(filler2)
        j2 += 1

# Reorder columns of both frames to match all_cols order
    left_copy = _reorder_columns(left_copy, all_cols)
    right_copy = _reorder_columns(right_copy, all_cols)

    return (left_copy, right_copy)


# Align row counts
fn align_rows(left: DataFrame, right: DataFrame) -> (DataFrame, DataFrame):
    """Ensure both frames have the same number of rows by padding with empty strings.
    The function returns copies and never mutates inputs.
    """
    var left_copy = left.copy()
    var right_copy = right.copy()

    var n_left = left_copy.nrows()
    var n_right = right_copy.nrows()
    if n_left == n_right:
        return (left_copy, right_copy)

    if n_left > n_right:
        var pad = n_left - n_right
        var j = 0
        while j < right_copy.ncols():
            var c = 0
            while c < pad:
                right_copy.cols[j].append(String(""))
                c += 1
            j += 1
    else:
        var pad2 = n_right - n_left
        var i = 0
        while i < left_copy.ncols():
            var c2 = 0
            while c2 < pad2:
                left_copy.cols[i].append(String(""))
                c2 += 1
            i += 1

    return (left_copy, right_copy)


# Internal: reorder columns to match a target order
fn _reorder_columns(df: DataFrame, order: List[String]) -> DataFrame:
    var out = DataFrame()
    out.index_name = df.index_name

# Build new columns in the exact order
    var i = 0
    while i < len(order):
        var name = order[i]
        var idx = -1
        var j = 0
        while j < len(df.col_names):
            if df.col_names[j] == name:
                idx = j
                break
            j += 1
        if idx == -1:
# Should not happen because caller guarantees presence;
# still, create an empty column of proper length defensively.
            var filler = List[String]()
            var r = 0
            while r < df.nrows():
                filler.append(String(""))
                r += 1
            out.col_names.append(name)
            out.cols.append(filler)
        else:
            out.col_names.append(name)
            out.cols.append(df.cols[idx])
        i += 1

    return out
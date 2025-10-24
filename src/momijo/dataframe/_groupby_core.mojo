# Project:      Momijo
# Module:       dataframe._groupby_core
# File:         _groupby_core.mojo
# Path:         dataframe/_groupby_core.mojo
#
# Description:  dataframe._groupby_core —  Groupby Core module for Momijo DataFrame.
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
#   - Key functions: _col_index, _safe_get_col, _make_key, _parse_float, _df_from, _as_str_col, groupby_agg, groupby_transform, left_join, __init__, make_module_state, groupby_sum_f64

from momijo.dataframe.frame import DataFrame
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_f64, col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.helpers import find_col, parse_f64_or_zero
from momijo.dataframe.series_bool import append

from math import sqrt
from collections import Dict, List

# Aggregations covered:
# mean, sum, min, max, count, std, var, median, first, last, any, all
struct Agg(Copyable, Movable):
    var tag: Int

    fn __init__(out self, tag: Int):
        self.tag = tag

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag

    # -------------------- private tag ids (via static funcs; no globals) --------------------
    @staticmethod
    fn _tag_mean() -> Int:   return 0
    @staticmethod
    fn _tag_sum() -> Int:    return 1
    @staticmethod
    fn _tag_min() -> Int:    return 2
    @staticmethod
    fn _tag_max() -> Int:    return 3
    @staticmethod
    fn _tag_count() -> Int:  return 4
    @staticmethod
    fn _tag_std() -> Int:    return 5
    @staticmethod
    fn _tag_var() -> Int:    return 6
    @staticmethod
    fn _tag_median() -> Int: return 7
    @staticmethod
    fn _tag_first() -> Int:  return 8
    @staticmethod
    fn _tag_last() -> Int:   return 9
    @staticmethod
    fn _tag_any() -> Int:    return 10
    @staticmethod
    fn _tag_all() -> Int:    return 11

    # -------------------- factories --------------------
    @staticmethod
    fn mean() -> Self:
        var a = Agg(Agg._tag_mean())
        return a.copy()

    @staticmethod
    fn sum() -> Self:
        var a = Agg(Agg._tag_sum())
        return a.copy()

    @staticmethod
    fn min() -> Self:
        var a = Agg(Agg._tag_min())
        return a.copy()

    @staticmethod
    fn max_() -> Self:
        var a = Agg(Agg._tag_max())
        return a.copy()

    @staticmethod
    fn count() -> Self:
        var a = Agg(Agg._tag_count())
        return a.copy()

    @staticmethod
    fn std() -> Self:
        var a = Agg(Agg._tag_std())
        return a.copy()

    @staticmethod
    fn var_() -> Self:
        var a = Agg(Agg._tag_var())
        return a.copy()

    @staticmethod
    fn median() -> Self:
        var a = Agg(Agg._tag_median())
        return a.copy()

    @staticmethod
    fn first() -> Self:
        var a = Agg(Agg._tag_first())
        return a.copy()

    @staticmethod
    fn last() -> Self:
        var a = Agg(Agg._tag_last())
        return a.copy()

    @staticmethod
    fn any() -> Self:
        var a = Agg(Agg._tag_any())
        return a.copy()

    @staticmethod
    fn all() -> Self:
        var a = Agg(Agg._tag_all())
        return a.copy()

    # -------------------- predicates --------------------
    fn is_mean(self) -> Bool:   return self.tag == Agg._tag_mean()
    fn is_sum(self) -> Bool:    return self.tag == Agg._tag_sum()
    fn is_min(self) -> Bool:    return self.tag == Agg._tag_min()
    fn is_max(self) -> Bool:    return self.tag == Agg._tag_max()
    fn is_count(self) -> Bool:  return self.tag == Agg._tag_count()
    fn is_std(self) -> Bool:    return self.tag == Agg._tag_std()
    fn is_var(self) -> Bool:    return self.tag == Agg._tag_var()
    fn is_median(self) -> Bool: return self.tag == Agg._tag_median()
    fn is_first(self) -> Bool:  return self.tag == Agg._tag_first()
    fn is_last(self) -> Bool:   return self.tag == Agg._tag_last()
    fn is_any(self) -> Bool:    return self.tag == Agg._tag_any()
    fn is_all(self) -> Bool:    return self.tag == Agg._tag_all()

    # -------------------- helpers --------------------
    fn name(self) -> String:
        if self.is_mean():   return String("mean")
        if self.is_sum():    return String("sum")
        if self.is_min():    return String("min")
        if self.is_max():    return String("max")
        if self.is_count():  return String("count")
        if self.is_std():    return String("std")
        if self.is_var():    return String("var")
        if self.is_median(): return String("median")
        if self.is_first():  return String("first")
        if self.is_last():   return String("last")
        if self.is_any():    return String("any")
        if self.is_all():    return String("all")
        return String("unknown")

    @staticmethod
    fn parse(name: String) -> Self:
        # Accept common aliases in lowercase
        var n = name.lower()
        if n == "mean" or n == "avg" or n == "average":
            return Agg.mean()
        if n == "sum" or n == "total":
            return Agg.sum()
        if n == "min" or n == "minimum":
            return Agg.min()
        if n == "max" or n == "maximum":
            return Agg.max_()
        if n == "count" or n == "size" or n == "len":
            return Agg.count()
        if n == "std" or n == "stdev" or n == "stddev":
            return Agg.std()
        if n == "var" or n == "variance":
            return Agg.var_()
        if n == "median" or n == "p50":
            return Agg.median()
        if n == "first" or n == "head":
            return Agg.first()
        if n == "last" or n == "tail":
            return Agg.last()
        if n == "any":
            return Agg.any()
        if n == "all":
            return Agg.all()
        var a = Agg(-1)
        return a.copy()

    fn __str__(self) -> String:
        return self.name()
# ------------------------ Utilities ------------------------

fn _col_index(df: DataFrame, name: String) -> Int:
    var i = 0
    while i < len(df.col_names):
        if df.col_names[i] == name:
            return i
        i += 1
    return -1
 



fn _parse_float(s: String, out out_ok: Bool) -> Float64:
# Minimal permissive parser: accepts optional sign and one dot.
# Returns 0.0 if parsing fails and sets out_ok=false.
    var has_digit = False
    var has_dot = False
    var neg = False
    var i = 0
    var acc: Float64 = 0.0
    var frac: Float64 = 0.0
    var denom: Float64 = 1.0
    if len(s) == 0:
        out_ok = False
        return 0.0
    if s[0] == String("-")[0]:
        neg = True
        i = 1
    elif s[0] == String("+")[0]:
        i = 1
    while i < len(s):
        var b = s[i]
        if b >= String("0")[0] and b <= String("9")[0]:
            has_digit = True
            if not has_dot:
                acc = acc * 10.0 + Float64(b - String("0")[0])
            else:
                denom = denom * 10.0
                frac = frac + Float64(b - String("0")[0]) / denom
        elif b == String(".")[0] and not has_dot:
            has_dot = True
        else:
            out_ok = False
            return 0.0
        i += 1
    var val = acc + frac
    if neg:
        val = -val
    out_ok = has_digit
    return val

# Build a DataFrame from names and string columns (same row count across columns).
fn _df_from(names: List[String], cols: List[List[String]]) -> DataFrame:
    var out = DataFrame()
    out.col_names = names
    out.cols = cols
    return out

# Single string column helper.
fn _as_str_col(name: String, values: List[String]) -> (String, List[String]):
    return (name, values)


# ------------------------ GroupBy: aggregate ------------------------
# aggs: list of (col_name, op), where op in {"count","nunique","first","last","min","max","sum","mean"}
# For numeric ops (sum/mean), non-numeric values are ignored.

fn groupby_agg(df: DataFrame, by: List[String], aggs: List[(String, String)]) -> DataFrame:
# Pre-fetch key columns
    var key_cols = List[List[String]]()
    var bi = 0
    while bi < len(by):
        key_cols.append(_safe_get_col(df, by[bi]))
        bi += 1

# Group map: key_string -> (first_row_index, size, list of row indices)
    var key_of_row = List[String]()
    key_of_row.reserve(df.nrows())

    var group_first = Dictionary[String, Int]()
    var group_size = Dictionary[String, Int]()
    var group_rows = Dictionary[String, List[Int]]()

    var r = 0
    while r < df.nrows():
# Build key values for row r
        var kv = List[String]()
        var kci = 0
        while kci < len(key_cols):
            kv.append(key_cols[kci][r])
            kci += 1
        var k = _make_key(kv)
        key_of_row.append(k)

        if not group_first.contains(k):
            group_first[k] = r
            group_size[k] = 1
            var lst = List[Int]()
            lst.append(r)
            group_rows[k] = lst
        else:
            group_size[k] = group_size[k] + 1
            var lst2 = group_rows[k]
            lst2.append(r)
            group_rows[k] = lst2
        r += 1

# Prepare output columns: keys first
    var out_names = List[String]()
    var out_cols = List[List[String]]()

# Materialize distinct groups in encounter order
    var keys_order = List[String]()
    var seen = Set[String]()
    var z = 0
    while z < len(key_of_row):
        var kk = key_of_row[z]
        if not seen.contains(kk):
            seen.insert(kk)
            keys_order.append(kk)
        z += 1

# Split keys into per-key columns
    var gcol_lists = List[List[String]]()
    var b2 = 0
    while b2 < len(by):
        gcol_lists.append(List[String]())
        b2 += 1

    var kidx = 0
    while kidx < len(keys_order):
        var key = keys_order[kidx]
# decompose key back to values
        var vals = List[String]()
        var cur = String("")
        var i = 0
        while i < len(key):
            var ch = key[i]
            if ch == String(UInt8(1)):
                vals.append(cur)
                cur = String("")
            else:
                cur += String(ch)
            i += 1
        vals.append(cur)
        var vi = 0
        while vi < len(vals):
            gcol_lists[vi].append(vals[vi])
            vi += 1
        kidx += 1

# Append key columns to output
    var bi2 = 0
    while bi2 < len(by):
        out_names.append(by[bi2])
        out_cols.append(gcol_lists[bi2])
        bi2 += 1

# Aggregations
    var agg_i = 0
    while agg_i < len(aggs):
        var pair = aggs[agg_i]
        var col_name = pair.0
        var op = pair.1

        var source = _safe_get_col(df, col_name)

        var out_col = List[String]()

        var g = 0
        while g < len(keys_order):
            var key = keys_order[g]
            var rows = group_rows[key]

            if op == String("count") or op == String("size"):
                out_col.append(String(len(rows)))
            elif op == String("nunique"):
                var uniq = Set[String]()
                var j = 0
                while j < len(rows):
                    var val = source[rows[j]]
                    uniq.insert(val)
                    j += 1
                out_col.append(String(len(uniq)))
            elif op == String("first") or op == String("head"):
                if len(rows) > 0:
                    out_col.append(source[rows[0]])
                else:
                    out_col.append(String(""))
            elif op == String("last") or op == String("tail"):
                if len(rows) > 0:
                    out_col.append(source[rows[len(rows)-1]])
                else:
                    out_col.append(String(""))
            elif op == String("min"):
                var v = String("")
                var init = False
                var j2 = 0
                while j2 < len(rows):
                    var s = source[rows[j2]]
                    if not init or s < v:
                        v = s
                        init = True
                    j2 += 1
                out_col.append(v)
            elif op == String("max"):
                var v2 = String("")
                var init2 = False
                var j3 = 0
                while j3 < len(rows):
                    var s2 = source[rows[j3]]
                    if not init2 or s2 > v2:
                        v2 = s2
                        init2 = True
                    j3 += 1
                out_col.append(v2)
            elif op == String("sum") or op == String("mean"):
                var total: Float64 = 0.0
                var cnt = 0
                var j4 = 0
                while j4 < len(rows):
                    var ok = False
                    var valf = _parse_float(source[rows[j4]], out_ok=ok)
                    if ok:
                        total = total + valf
                        cnt = cnt + 1
                    j4 += 1
                if op == String("sum"):
                    out_col.append(String(total))
                else:
                    if cnt == 0:
                        out_col.append(String("0"))
                    else:
                        out_col.append(String(total / Float64(cnt)))
            else:
# Unknown op: fill empty
                out_col.append(String(""))
            g += 1

        out_names.append(col_name + String("_") + op)
        out_cols.append(out_col)
        agg_i += 1

    return _df_from(out_names, out_cols)


# ------------------------ GroupBy: transform ------------------------
# Returns a per-row vector aligned with df.nrows().
# Supported ops: "count", "nunique" (per-group), "rank" (1..n within group), "cumcount" (0..n-1)

fn _make_key(vals: List[String]) -> String:
    var out = String("")
    var i = 0
    while i < len(vals):
        if i > 0:
            out += String(UInt8(1))  # separator unlikely in real data
        out += vals[i]
        i += 1
    return out


# Helper: get column safely
fn _safe_get_col(df: DataFrame, col: String) -> List[String]:
    var cidx = -1
    var j = 0
    var ncols = len(df.col_names)
    while j < ncols:
        if df.col_names[j] == col:
            cidx = j
            break
        j += 1
    var out: List[String] = List[String]()
    if cidx == -1:
        return out.copy()
    var r = 0
    var nrows = df.nrows()
    while r < nrows:
        out.append(df.cols[cidx].value_str(r))
        r += 1
    return out.copy()


fn groupby_transform(df: DataFrame, by: List[String], col: String, op: String) -> List[Float64]:
    var out: List[Float64] = List[Float64]()
    out.reserve(df.nrows())

    var key_cols: List[List[String]] = List[List[String]]()
    var bidx = 0
    while bidx < len(by):
        key_cols.append(_safe_get_col(df, by[bidx]))
        bidx += 1

    var key_of_row: List[String] = List[String]()
    var r = 0
    while r < df.nrows():
        var kv: List[String] = List[String]()
        var kci = 0
        while kci < len(key_cols):
            kv.append(key_cols[kci][r])
            kci += 1
        key_of_row.append(_make_key(kv))
        r += 1

    var source: List[String] = _safe_get_col(df, col)

    if op == "count" or op == "size":
        var counts: Dict[String, Int] = Dict[String, Int]()
        var i = 0
        while i < len(key_of_row):
            var k = key_of_row[i]
            var c: Int
            try:
                c = counts[k]
            except:
                c = 0
            counts[k] = c + 1
            i += 1
        i = 0
        while i < len(key_of_row):
            var k = key_of_row[i]
            var v: Int
            try:
                v = counts[k]
            except:
                v = 0
            out.append(Float64(v))
            i += 1
        return out.copy()

    if op == "nunique" or op == "distinct":
        var sets: Dict[String, Dict[String, Bool]] = Dict[String, Dict[String, Bool]]()
        var i = 0
        while i < len(key_of_row):
            var k = key_of_row[i]
            var tmp: Dict[String, Bool]
            try:
                tmp = sets[k].copy()
            except:
                tmp = Dict[String, Bool]()
            tmp[source[i]] = True
            sets[k] = tmp.copy()
            i += 1
        i = 0
        while i < len(key_of_row):
            var k = key_of_row[i]
            var sz: Int
            try:
                sz = len(sets[k])
            except:
                sz = 0
            out.append(Float64(sz))
            i += 1
        return out.copy()

    if op == "rank" or op == "cumcount":
        var counters: Dict[String, Int] = Dict[String, Int]()
        var i = 0
        while i < len(key_of_row):
            var k = key_of_row[i]
            var c: Int
            try:
                c = counters[k]
            except:
                c = 0
            if op == "rank":
                out.append(Float64(c + 1))
            else:
                out.append(Float64(c))
            counters[k] = c + 1
            i += 1
        return out.copy()

    if op == "zscore":
        var sum: Dict[String, Float64] = Dict[String, Float64]()
        var cnt: Dict[String, Int] = Dict[String, Int]()
        var sq: Dict[String, Float64] = Dict[String, Float64]()
        var i = 0
        while i < len(source):
            var key = key_of_row[i]
            var v: Float64
            try:
                v = Float64(source[i])
            except:
                i += 1
                continue
            var s: Float64
            var n: Int
            var s2: Float64
            try:
                s = sum[key]
                n = cnt[key]
                s2 = sq[key]
            except:
                s = 0.0
                n = 0
                s2 = 0.0
            sum[key] = s + v
            cnt[key] = n + 1
            sq[key] = s2 + v * v
            i += 1
        i = 0
        while i < len(source):
            var key = key_of_row[i]
            var v: Float64
            try:
                v = Float64(source[i])
            except:
                out.append(0.0)
                i += 1
                continue
            var s: Float64
            var n: Int
            var s2: Float64
            try:
                s = sum[key]
                n = cnt[key]
                s2 = sq[key]
            except:
                s = 0.0
                n = 0
                s2 = 0.0
            var m: Float64
            if n > 0:
                m = s / Float64(n)
            else:
                m = 0.0
            var varg: Float64
            if n > 0:
                varg = s2 / Float64(n) - m * m
            else:
                varg = 0.0
            if varg < 0.0:
                varg = 0.0
            var z: Float64
            if varg > 0.0:
                z = (v - m) / sqrt(varg)
            else:
                z = (v - m) / 1.0
            out.append(z)
            i += 1
        return out.copy()

    var i = 0
    while i < df.nrows():
        out.append(0.0)
        i += 1
    return out.copy()








struct ModuleState:
    var keys
    fn __init__(out self, keys):
        self.keys = keys

fn make_module_state(state) -> ModuleState:
    return ModuleState(List[String]())


    var sums = List[Float64]()

    var i = 0
    while i < df.height(, state: ModuleState):
        var k = df.cols[ik][i]
        var v = parse_f64_or_zero(df.cols[iv][i])
# find slot
        var pos = -1
        var j = 0
        while j < len(keys):
            if keys[j] == k:
                pos = j
            j += 1
        if pos == -1:
            keys.append(k)
            sums.append(0.0)
            pos = len(keys) - 1
        sums[pos] = sums[pos] + v
        i += 1

    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, keys),
                                 col_f64(String("sum_") + val, sums)]))
fn groupby_sum_f64(df: DataFrame, key: String, val: String) -> DataFrame
    var ik = find_col(df, key)
    var iv = find_col(df, val)
    var keys = List[String]()
    var sums = List[Float64]()
    var i = 0
    while i < df.nrows():
        var k = df.cols[ik][i]
        var v = parse_f64_or_zero(df.cols[iv][i])
# find slot
        var pos = -1
        var j = 0
        while j < len(state.keys):
            if state.keys[j] == k:
                pos = j
            j += 1
        if pos == -1:
            state.keys.append(k)
            sums.append(0.0)
            pos = len(state.keys) - 1
        sums[pos] = sums[pos] + v
        i += 1

    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, state.keys),
                                 col_f64(String("sum_") + val, sums)]))

 

# Aggregate functions
fn apply_agg(values: List[Optional[Float64]], func: String) -> Optional[Float64]:
    if len(values) == 0:
        return None
    var clean_values: List[Float64] = List[Float64]()
    for v in values:
        if v is not None:
            clean_values.append(v.value())
    if len(clean_values) == 0:
        return None
    if func == "sum":
        var s: Float64 = 0.0
        for v in clean_values:
            s += v
        return s
    elif func == "mean":
        var s: Float64 = 0.0
        for v in clean_values:
            s += v
        return s / Float64(len(clean_values))
    elif func == "min":
        var m = clean_values[0]
        for v in clean_values[1:]:
            if v < m:
                m = v
        return m
    elif func == "max":
        var m = clean_values[0]
        for v in clean_values[1:]:
            if v > m:
                m = v
        return m
    else:
        raise Error("Unsupported aggregation function")

fn _is_digit(c: StringSlice) -> Bool:
    var digits = String("0123456789")
    var j = 0
    var m = len(digits)
    while j < m:
        if c == digits[j]:
            return True
        j += 1
    return False

fn _digit_val(c: StringSlice) -> Float64:
    var digits = String("0123456789")
    var j = 0
    var m = len(digits)
    while j < m:
        if c == digits[j]:
            return Float64(j)
        j += 1
    return 0.0

fn _parse_f64_safe(s: String) -> Float64:
    var neg = False
    var i = 0
    var n = len(s)
    if n == 0:
        return 0.0
    if s[0] == "-"[0]:
        neg = True
        i = 1
    var int_part: Float64 = 0.0
    var frac_part: Float64 = 0.0
    var frac_div: Float64 = 1.0
    var seen_dot = False
    while i < n:
        var c = s[i]
        if c == "."[0] and not seen_dot:
            seen_dot = True
            i += 1
            continue
        if not _is_digit(c):
            break
        var d = _digit_val(c)
        if not seen_dot:
            int_part = int_part * 10.0 + d
        else:
            frac_div = frac_div * 10.0
            frac_part = frac_part + d / frac_div
        i += 1
    var out = int_part + frac_part
    if neg:
        out = -out
    return out



fn _apply_agg_local_f64(vals: List[Float64], func: String) -> Float64:
    var m: Float64 = 0.0
    var n = len(vals)
    if func == String("count"):
        return Float64(n)
    if n == 0:
        return 0.0
    var i = 0
    if func == String("max"):
        m = vals[0]
        i = 1
        while i < n:
            if vals[i] > m:
                m = vals[i]
            i += 1
        return m
    if func == String("min"):
        m = vals[0]
        i = 1
        while i < n:
            if vals[i] < m:
                m = vals[i]
            i += 1
        return m
    if func == String("mean"):
        var s: Float64 = 0.0
        while i < n:
            s += vals[i]
            i += 1
        return s / Float64(n)
    return 0.0

fn groupby(frame: DataFrame, by: List[String], aggs: Dict[String, List[String]]) -> DataFrame:
    # 1) Resolve group-by columns
    var gb_cols = List[Column]()
    var bi = 0
    while bi < len(by):
        gb_cols.append(frame.get_column(by[bi]))
        bi += 1

    var n = frame.nrows()

    # 2) Build groups: key -> row indices, and remember key parts
    var groups = Dict[String, List[Int]]()
    var key_parts_map = Dict[String, List[String]]()

    var r = 0
    while r < n:
        var parts = List[String]()
        var i = 0
        while i < len(gb_cols):
            parts.append(gb_cols[i].get_string(r))
            i += 1

        # join with a safe delimiter
        var key = String("")
        var kpi = 0
        while kpi < len(parts):
            if kpi > 0:
                key += String("|")
            key += parts[kpi]
            kpi += 1

        # groups[key].append(r)
        var opt_idx_list = groups.get(key)
        var idx_list = List[Int]()
        if opt_idx_list is not None:
            idx_list = opt_idx_list.value().copy()
        idx_list.append(r)
        groups[key] = idx_list.copy()

        # store key parts once
        if key_parts_map.get(key) is None:
            key_parts_map[key] = parts.copy()

        r += 1

    # 3) Build output column order: by-cols then "<col>_<op>" in stable name order
    var col_order = List[String]()
    var j = 0
    while j < len(by):
        col_order.append(by[j])
        j += 1

    var agg_keys = List[String]()
    for k in aggs.keys():
        agg_keys.append(String(k))
    var a = 0
    while a + 1 < len(agg_keys):
        var b = a + 1
        while b < len(agg_keys):
            if agg_keys[b] < agg_keys[a]:
                var tmp = agg_keys[a]
                agg_keys[a] = agg_keys[b]
                agg_keys[b] = tmp
            b += 1
        a += 1

    var ak = 0
    while ak < len(agg_keys):
        var src = agg_keys[ak]
        var funcs_opt = aggs.get(src)
        if funcs_opt is not None:
            var fs = funcs_opt.value().copy()
            var fi = 0
            while fi < len(fs):
                col_order.append(String(src + "_" + fs[fi]))
                fi += 1
        ak += 1

    # 4) Prepare output buckets
    var out_cols = Dict[String, List[String]]()
    var co = 0
    while co < len(col_order):
        out_cols[col_order[co]] = List[String]()
        co += 1

    # stable iteration over groups
    var keys = List[String]()
    for k in groups.keys():
        keys.append(String(k))
    var i1 = 0
    while i1 + 1 < len(keys):
        var i2 = i1 + 1
        while i2 < len(keys):
            if keys[i2] < keys[i1]:
                var tk = keys[i1]
                keys[i1] = keys[i2]
                keys[i2] = tk
            i2 += 1
        i1 += 1

    # 5) Emit rows
    var kk = 0
    while kk < len(keys):
        var key = keys[kk]
        var idxs = groups.get(key).value().copy()
        var gvals = key_parts_map.get(key).value().copy()

        # emit group-by columns
        var gi = 0
        while gi < len(by):
            var cname = by[gi]
            var cur = out_cols.get(cname).value().copy()
            cur.append(gvals[gi])
            out_cols[cname] = cur.copy()
            gi += 1

        # aggregated columns
        var ak2 = 0
        while ak2 < len(agg_keys):
            var col_name = agg_keys[ak2]
            var fs_opt = aggs.get(col_name)
            if fs_opt is not None:
                var fs = fs_opt.value().copy()

                # fetch column
                var col = frame.get_column(col_name)

                # collect both string and numeric (parsed) views
                var vals_str = List[String]()
                var vals_num = List[Float64]()
                var has_num = List[Bool]()
                var t = 0
                while t < len(idxs):
                    var idx = idxs[t]
                    var sv = col.get_string(idx)
                    vals_str.append(sv)
                    var x = _parse_f64_safe(sv)
                    # decide if numeric by round-trip check
                    # (simple heuristic: if parse yields NaN or sv isn't numeric-looking, mark false)
                    var ok = _is_numeric_like(sv)
                    has_num.append(ok)
                    if ok:
                        vals_num.append(x)
                    else:
                        # keep length parity (optional)
                        vals_num.append(0.0)
                    t += 1

                var fi2 = 0
                while fi2 < len(fs):
                    var func = fs[fi2]
                    var out_name = String(col_name + "_" + func)

                    var out_list = out_cols.get(out_name).value().copy()

                    if func == "count" or func == "size":
                        out_list.append(String(len(idxs)))
                    elif func == "nunique" or func == "distinct":
                        # unique over STRING values
                        var seen = Dict[String, Bool]()
                        var u = 0
                        while u < len(vals_str):
                            seen[vals_str[u]] = True
                            u += 1
                        out_list.append(String(len(seen)))
                    elif func == "first":
                        if len(vals_str) > 0:
                            out_list.append(vals_str[0])
                        else:
                            out_list.append(String(""))
                    elif func == "last":
                        if len(vals_str) > 0:
                            out_list.append(vals_str[len(vals_str) - 1])
                        else:
                            out_list.append(String(""))

                    elif func == "min" or func == "max" or func == "sum" or func == "mean":
                        # numeric-only ops; ignore non-numeric entries
                        var cnt: Int = 0
                        var s: Float64 = 0.0
                        var mn: Float64 = 0.0
                        var mx: Float64 = 0.0
                        var any = False

                        var q = 0
                        while q < len(vals_num):
                            if has_num[q]:
                                var v = vals_num[q]
                                if not any:
                                    any = True
                                    mn = v
                                    mx = v
                                else:
                                    if v < mn: mn = v
                                    if v > mx: mx = v
                                s += v
                                cnt += 1
                            q += 1

                        if func == "min":
                            out_list.append(String(mn if any else 0.0))
                        elif func == "max":
                            out_list.append(String(mx if any else 0.0))
                        elif func == "sum":
                            out_list.append(String(s))
                        else:  # mean
                            out_list.append(String(s / Float64(cnt) if cnt > 0 else 0.0))
                    else:
                        # unknown op → empty/default
                        out_list.append(String(""))

                    out_cols[out_name] = out_list.copy()
                    fi2 += 1
            ak2 += 1

        kk += 1

    # 6) Materialize DataFrame
    var rp = make_pairs()
    var oi = 0
    while oi < len(col_order):
        var cname = col_order[oi]
        var arr = out_cols.get(cname).value().copy()
        rp = pairs_append(rp, cname, arr)
        oi += 1

    var out = df_from_pairs(rp)
    return out


# Heuristic numeric check that avoids per-character string scanning.
# Treats "0"/"0.0"/"-0"/"-0.0" as numeric; otherwise uses parse result.
fn _is_numeric_like(s: String) -> Bool:
    if len(s) == 0:
        return False
    if s == String("0") or s == String("0.0") or s == String("-0") or s == String("-0.0"):
        return True
    var v = _parse_f64_safe(s)
    # If parsing produced non-zero, assume it's a valid numeric string.
    if v != 0.0:
        return True
    # v == 0.0 here: only accept if the literal is another common zero form.
    return s == String("+0") or s == String("+0.0")


# returns column index or -1
fn _find_col_index(frame: DataFrame, name: String) -> Int:
    var j = 0
    var n = len(frame.col_names)
    while j < n:
        if frame.col_names[j] == name:
            return j
        j += 1
    return -1

# safe parse float from string; returns Optional[Float64]
fn _try_f64(s: String) -> Optional[Float64]:
    var n = len(s)
    if n == 0:
        var none: Optional[Float64] = None
        return none
    # minimal parser: allow leading/trailing spaces
    var ok = False
    var acc = 0.0
    var sign = 1.0
    var i = 0
    # skip spaces
    while i < n and (ord(s[i]) == 32 or ord(s[i]) == 9 or ord(s[i]) == 10 or ord(s[i]) == 13):
        i += 1
    if i >= n:
        var none2: Optional[Float64] = None
        return none2
    # sign
    if s[i] == "-" or s[i] == "+":
        if s[i] == "-":
            sign = -1.0
        i += 1
        if i >= n:
            var none3: Optional[Float64] = None
            return none3
    # integer part
    var int_any = False
    while i < n and (ord(s[i]) >= ord("0") and ord(s[i]) <= ord("9")):
        acc = acc * 10.0 + Float64(ord(s[i]) - ord("0"))
        i += 1
        int_any = True
    # fraction
    if i < n and s[i] == ".":
        i += 1
        var base = 0.1
        var frac_any = False
        while i < n and (ord(s[i]) >= ord("0") and ord(s[i]) <= ord("9")):
            acc = acc + base * Float64(ord(s[i]) - ord("0"))
            base = base * 0.1
            i += 1
            frac_any = True
        ok = int_any or frac_any
    else:
        ok = int_any
    # skip trailing spaces
    while i < n and (ord(s[i]) == 32 or ord(s[i]) == 9 or ord(s[i]) == 10 or ord(s[i]) == 13):
        i += 1
    if i != n:
        var none4: Optional[Float64] = None
        return none4
    if not ok:
        var none5: Optional[Float64] = None
        return none5
    var res: Optional[Float64] = None
    res = sign * acc
    return res



fn pivot_table(frame: DataFrame,
               index: String,
               columns: String,
               values: String,
               agg: Agg,
               margins: Bool,
               margins_name: String) -> DataFrame:
    var idx_i = _find_col_index(frame, index)
    var col_i = _find_col_index(frame, columns)
    var val_i = _find_col_index(frame, values)
    if idx_i < 0 or col_i < 0 or val_i < 0:
        return frame.copy()

    var rows_unique = List[String]()
    var cols_unique = List[String]()

    var n = frame.nrows()
    var r = 0
    while r < n:
        var rv = String(frame.cols[idx_i].get_string(r))
        var cv = String(frame.cols[col_i].get_string(r))
        var seen_r = False
        var i = 0
        while i < len(rows_unique):
            if rows_unique[i] == rv:
                seen_r = True
                break
            i += 1
        if not seen_r:
            rows_unique.append(rv)
        var seen_c = False
        var j = 0
        while j < len(cols_unique):
            if cols_unique[j] == cv:
                seen_c = True
                break
            j += 1
        if not seen_c:
            cols_unique.append(cv)
        r += 1

    var sums = List[List[Float64]]()
    var counts = List[List[Int]]()
    var i2 = 0
    while i2 < len(rows_unique):
        var row_s = List[Float64]()
        var row_c = List[Int]()
        var j2 = 0
        while j2 < len(cols_unique):
            row_s.append(0.0)
            row_c.append(0)
            j2 += 1
        sums.append(row_s.copy())
        counts.append(row_c.copy())
        i2 += 1

    var rr = 0
    while rr < n:
        var rv = String(frame.cols[idx_i].get_string(rr))
        var cv = String(frame.cols[col_i].get_string(rr))
        var vv = String(frame.cols[val_i].get_string(rr))

        var rpos = 0
        while rpos < len(rows_unique) and rows_unique[rpos] != rv:
            rpos += 1
        var cpos = 0
        while cpos < len(cols_unique) and cols_unique[cpos] != cv:
            cpos += 1

        if rpos < len(rows_unique) and cpos < len(cols_unique):
            var fopt = _try_f64(vv)
            if not (fopt is None):
                sums[rpos][cpos] = sums[rpos][cpos] + fopt.value()
                counts[rpos][cpos] = counts[rpos][cpos] + 1
        rr += 1

    var out_names = List[String]()
    var out_cols = List[List[String]]()

    out_names.append(index)
    out_cols.append(List[String]())

    var cj = 0
    while cj < len(cols_unique):
        out_names.append(cols_unique[cj])
        out_cols.append(List[String]())
        cj += 1

    if margins:
        out_names.append(margins_name)
        out_cols.append(List[String]())

    var ri = 0
    while ri < len(rows_unique):
        out_cols[0].append(rows_unique[ri])

        var row_sum = 0.0
        var row_cnt = 0

        var cj2 = 0
        while cj2 < len(cols_unique):
            var mstr = String("")
            if counts[ri][cj2] > 0:
                var m = sums[ri][cj2] / Float64(counts[ri][cj2])
                mstr = String(m)
                row_sum = row_sum + sums[ri][cj2]
                row_cnt = row_cnt + counts[ri][cj2]
            out_cols[1 + cj2].append(mstr)
            cj2 += 1

        if margins:
            var rmean = String("")
            if row_cnt > 0:
                rmean = String(row_sum / Float64(row_cnt))
            out_cols[1 + len(cols_unique)].append(rmean)

        ri += 1

    if margins:
        out_cols[0].append(margins_name)

        var col_idx = 0
        while col_idx < len(cols_unique):
            var col_sum = 0.0
            var col_cnt = 0
            var rx = 0
            while rx < len(rows_unique):
                col_sum = col_sum + sums[rx][col_idx]
                col_cnt = col_cnt + counts[rx][col_idx]
                rx += 1
            var cmean = String("")
            if col_cnt > 0:
                cmean = String(col_sum / Float64(col_cnt))
            out_cols[1 + col_idx].append(cmean)
            col_idx += 1

        var all_sum = 0.0
        var all_cnt = 0
        var rx2 = 0
        while rx2 < len(rows_unique):
            var cx2 = 0
            while cx2 < len(cols_unique):
                all_sum = all_sum + sums[rx2][cx2]
                all_cnt = all_cnt + counts[rx2][cx2]
                cx2 += 1
            rx2 += 1
        var all_mean = String("")
        if all_cnt > 0:
            all_mean = String(all_sum / Float64(all_cnt))
        out_cols[1 + len(cols_unique)].append(all_mean)

    var pairs = make_pairs()
    var k = 0
    while k < len(out_names):
        pairs = pairs_append(pairs, out_names[k], out_cols[k])
        k += 1
    var out = df_from_pairs(pairs)
    return out


@always_inline
fn _fill_or_empty(fill_value: Optional[String]) -> String:
    if fill_value is None:
        return String("")
    return fill_value.value()

fn pivot_table(frame: DataFrame,
               index: String,
               columns: String,
               values: String,
               agg: Agg,
               margins: Bool,
               margins_name: String,
               fill_value: Optional[String] = None) -> DataFrame:
    var idx_i = _find_col_index(frame, index)
    var col_i = _find_col_index(frame, columns)
    var val_i = _find_col_index(frame, values)
    if idx_i < 0 or col_i < 0 or val_i < 0:
        return frame.copy()

    var rows_unique = List[String]()
    var cols_unique = List[String]()

    var n = frame.nrows()
    var r = 0
    while r < n:
        var rv = String(frame.cols[idx_i].get_string(r))
        var cv = String(frame.cols[col_i].get_string(r))
        var seen_r = False
        var i = 0
        while i < len(rows_unique):
            if rows_unique[i] == rv:
                seen_r = True
                break
            i += 1
        if not seen_r:
            rows_unique.append(rv)

        var seen_c = False
        var j = 0
        while j < len(cols_unique):
            if cols_unique[j] == cv:
                seen_c = True
                break
            j += 1
        if not seen_c:
            cols_unique.append(cv)
        r += 1

    var sums = List[List[Float64]]()
    var counts = List[List[Int]]()
    var i2 = 0
    while i2 < len(rows_unique):
        var row_s = List[Float64]()
        var row_c = List[Int]()
        var j2 = 0
        while j2 < len(cols_unique):
            row_s.append(0.0)
            row_c.append(0)
            j2 += 1
        sums.append(row_s.copy())
        counts.append(row_c.copy())
        i2 += 1

    var rr = 0
    while rr < n:
        var rv = String(frame.cols[idx_i].get_string(rr))
        var cv = String(frame.cols[col_i].get_string(rr))
        var vv = String(frame.cols[val_i].get_string(rr))

        var rpos = 0
        while rpos < len(rows_unique) and rows_unique[rpos] != rv:
            rpos += 1
        var cpos = 0
        while cpos < len(cols_unique) and cols_unique[cpos] != cv:
            cpos += 1

        if rpos < len(rows_unique) and cpos < len(cols_unique):
            var fopt = _try_f64(vv)
            if not (fopt is None):
                sums[rpos][cpos] = sums[rpos][cpos] + fopt.value()
                counts[rpos][cpos] = counts[rpos][cpos] + 1
        rr += 1

    var out_names = List[String]()
    var out_cols = List[List[String]]()

    out_names.append(index)
    out_cols.append(List[String]())

    var cj = 0
    while cj < len(cols_unique):
        out_names.append(cols_unique[cj])
        out_cols.append(List[String]())
        cj += 1

    if margins:
        out_names.append(margins_name)
        out_cols.append(List[String]())

    var ri = 0
    while ri < len(rows_unique):
        out_cols[0].append(rows_unique[ri])

        var row_sum = 0.0
        var row_cnt = 0

        var cj2 = 0
        while cj2 < len(cols_unique):
            var mstr = String("")
            if counts[ri][cj2] > 0:
                # NOTE: current behavior = mean; extend with 'agg' later if needed
                var m = sums[ri][cj2] / Float64(counts[ri][cj2])
                mstr = String(m)
                row_sum = row_sum + sums[ri][cj2]
                row_cnt = row_cnt + counts[ri][cj2]
            else:
                mstr = _fill_or_empty(fill_value)
            out_cols[1 + cj2].append(mstr)
            cj2 += 1

        if margins:
            var rmean = String("")
            if row_cnt > 0:
                rmean = String(row_sum / Float64(row_cnt))
            else:
                rmean = _fill_or_empty(fill_value)
            out_cols[1 + len(cols_unique)].append(rmean)

        ri += 1

    if margins:
        out_cols[0].append(margins_name)

        var col_idx = 0
        while col_idx < len(cols_unique):
            var col_sum = 0.0
            var col_cnt = 0
            var rx = 0
            while rx < len(rows_unique):
                col_sum = col_sum + sums[rx][col_idx]
                col_cnt = col_cnt + counts[rx][col_idx]
                rx += 1
            var cmean = String("")
            if col_cnt > 0:
                cmean = String(col_sum / Float64(col_cnt))
            else:
                cmean = _fill_or_empty(fill_value)
            out_cols[1 + col_idx].append(cmean)
            col_idx += 1

        var all_sum = 0.0
        var all_cnt = 0
        var rx2 = 0
        while rx2 < len(rows_unique):
            var cx2 = 0
            while cx2 < len(cols_unique):
                all_sum = all_sum + sums[rx2][cx2]
                all_cnt = all_cnt + counts[rx2][cx2]
                cx2 += 1
            rx2 += 1
        var all_mean = String("")
        if all_cnt > 0:
            all_mean = String(all_sum / Float64(all_cnt))
        else:
            all_mean = _fill_or_empty(fill_value)
        out_cols[1 + len(cols_unique)].append(all_mean)

    var pairs = make_pairs()
    var k = 0
    while k < len(out_names):
        pairs = pairs_append(pairs, out_names[k], out_cols[k])
        k += 1
    var out = df_from_pairs(pairs)
    return out


fn pivot_table(frame: DataFrame,
               index: List[String],
               columns: List[String],
               values: String,
               agg: Agg,
               fill_value: Optional[String] = None,
               margins: Bool = False,
               margins_name: String = String("Total")) -> DataFrame:
    var idx = String("")
    if len(index) > 0: idx = index[0]
    var cols = String("")
    if len(columns) > 0: cols = columns[0]
    return pivot_table(frame, idx, cols, values, agg, margins, margins_name, fill_value)
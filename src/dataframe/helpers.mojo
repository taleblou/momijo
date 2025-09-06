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
# File: src/momijo/dataframe/helpers.mojo

#   from momijo.dataframe.column import Options
#   from momijo.dataframe.options import Options
# SUGGEST (alpha): from momijo.dataframe.column import Options
#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.arrow_core.array_stats import count
#   from momijo.core.ndarray import count
#   from momijo.core.shape import count
# SUGGEST (alpha): from momijo.arrow_core.array_stats import count
#   from momijo.dataframe.diagnostics import df_cell
#   from momijo.dataframe.frame import df_cell
# SUGGEST (alpha): from momijo.dataframe.diagnostics import df_cell
#   from momijo.dataframe.api import df_make
#   from momijo.dataframe.frame import df_make
# SUGGEST (alpha): from momijo.dataframe.api import df_make
#   from momijo.dataframe.column import dtype_name_of_column
#   from momijo.dataframe.diagnostics import dtype_name_of_column
# SUGGEST (alpha): from momijo.dataframe.column import dtype_name_of_column
#   from momijo.arrow_core.buffer import fill
#   from momijo.core.ndarray import fill
# SUGGEST (alpha): from momijo.arrow_core.buffer import fill
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
# NOTE: Migrated: original 'fn cumsum' renamed to 'legacy_cumsum'; 'cumsum' now imported from stdlib
from momijo.extras.stubs import Char, Comment, Copyright, IndexSlice, MIT, Migrated, Resample, SUGGEST, Strict, _once, _to_string, acc, additions, agg, and, basics, best_t, break, builder, bytes, cnt, contains, dense_f64, df_cell, diff_str, display_rows, else, f64, f_last_before, find, find_bool, find_int, find_next_quote, floor, frac_part, from, get_bool, get_column_at, get_f64, get_i64, get_string, getters, hdr, header_bytes, height, helpers, hlen, https, idx, idxs, if, indicator, int_part, is_f64, is_i64, ke, key, len, length, limit, line, lists, loc, match, max_rows, memory_usage, momijo, names, neg, next_f64, next_u32, not, observed, ok, or, ord, orders, original, pad, parser, print, remove_unused, remove_unused_categories, reorder, rep, replace, return, rnk, row, rows, seen, span, src, start, str_to_bytes, string, sx, t64, to_period, top, try, ty, val, validate, value, variants, wrappers, xs, xt_u32, ype
from momijo.dataframe.diagnostics import header
from momijo.dataframe.io_facade import read_csv
from momijo.dataframe.api import df_make
from momijo.tensor.indexing import slice
from momijo.dataframe.series_bool import append
from momijo.dataframe.stats_core import sqrt_f64
from momijo.dataframe.take import take_bool
from momijo.dataframe.take import take_str
from momijo.dataframe.take import take_f64
from momijo.dataframe.take import take_i64
from momijo.dataframe.sorting import argsort_str2
from momijo.dataframe.string_ops import compare_str_eq
from momijo.dataframe.index import rename_axis_demo
from momijo.dataframe.sorting import sort_values_key
from momijo.dataframe.rolling import rolling_apply_abs
from momijo.dataframe.timeseries import resample_sum_min
from momijo.dataframe.interval import searchsorted_f64
from momijo.dataframe.options import with_options
from momijo.dataframe.stats_core import corr_f64
from momijo.dataframe.stats_core import cumsum_i64
from momijo.dataframe.rolling import rolling_mean
from momijo.dataframe.string_ops import str_strip
from momijo.dataframe.string_ops import str_split_once
from momijo.dataframe.string_ops import str_contains
from momijo.dataframe.timezone import tz_convert
from momijo.dataframe.timezone import tz_localize_utc
from momijo.dataframe.sorting import rank_dense_f64
from momijo.dataframe.stats_transform import str_replace_all
from momijo.dataframe.stats_transform import value_counts_str
from momijo.dataframe.masks import isin_string
from momijo.dataframe.masks import between_i64
from momijo.dataframe.selection import iloc_col
from momijo.dataframe.selection import iloc_row
from momijo.dataframe.interval import Interval
from momijo.tensor.random import RNG
from momijo.dataframe.index import SimpleIndex
from momijo.dataframe.sampling import LCG, __init__
from momijo.dataframe.index import SimpleMultiIndex
from momijo.dataframe.expanding import expanding_corr_last
from momijo.dataframe.encoding import value_counts_bins
from momijo.dataframe.describe import percentile_f64
from momijo.dataframe.datetime_ops import dt_normalize
from momijo.dataframe.categorical import Categorical
from momijo.dataframe.asof import asof_last_before
from momijo.dataframe.aliases import value_counts
from momijo.arrow_core.array_stats import count, max_f64
from momijo.arrow_core.array_stats import min_f64
from momijo.arrow_core.array_stats import max
from momijo.arrow_core.array_stats import min
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
import pathlib
from momijo.dataframe.logical_plan import join
#   from momijo.dataframe.diagnostics import join_names
#   from momijo.dataframe.join import join_names
# SUGGEST (alpha): from momijo.dataframe.diagnostics import join_names
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
from momijo.dataframe.column import Options, col_i64, col_str, dtype_name_of_column, name
#   from momijo.core.result import ok
#   from momijo.tensor.errors import ok
# SUGGEST (alpha): from momijo.core.result import ok
#   from momijo.dataframe.io_csv import read_csv
#   from momijo.dataframe.io_facade import read_csv
# SUGGEST (alpha): from momijo.dataframe.io_csv import read_csv
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
#   from momijo.core.config import validate
#   from momijo.core.parameter import validate
#   from momijo.core.shape import validate
#   from momijo.core.types import validate
# SUGGEST (alpha): from momijo.core.config import validate
from momijo.dataframe.frame import

import col_str
from momijo.arr

ing
from momijo.dataframe.join import join_names, left_join
from momijo.dataframe.frame import DataFrame, ncols
from momijo.dataframe.frame import nrows
from momijo.core.parameter import state
from momijo.dataframe.frame import width
from momijo.dataframe.io_csv import is_bool, write_csv
import momijo.dataframe

# ---------------------------
# Validation utilities
# ---------------------------

# Raise if the DataFrame is empty by columns or rows.

struct SimpleMultiIndex:
    var level1: List[String]
    var level2: List[String]

struct LCG:
    var state: UInt64

fn preview_string(df: momijo.dataframe.DataFrame, max_rows: Int = 5) -> String
    var out = String("DataFrame(rows=" + String(df.height()) + ", cols=" + String(df.width()) + ")\n")

    # column names
    out += "Columns: ["
    var i = 0
    var w = df.width()
    while i < w:
        out += df.names[i]
        if i + 1 < w:
            out += ", "
        i += 1
    out += "]\n"

    # first rows
    var h = df.height()
    var r = 0
    while r < h and r < max_rows:
        out += "Row " + String(r) + ": "
        var c = 0
        while c < w:
            out += "<val>"          # TODO: replace with real Column getters
            if c + 1 < w:
                out += ", "
            c += 1
        out += "\n"
        r += 1

    if h > max_rows:
        out += "... (" + String(h - max_rows) + " more rows)\n"

    return out

# ---------------------------
# Strict (exception-throwing) wrappers
# ---------------------------

# Read CSV and throw if parsing yields an empty DataFrame.

fn try_read_csv(path: String, out df: out momijo.dataframe.DataFrame) -> Bool
    df = momijo.dataframe.read_csv(path)
    if df.width() == 0 or df.height() == 0:
        # normalize to empty result on failure/emptiness
        df = momijo.dataframe.DataFrame()
        return false
    return true

# Read CSV or return the provided fallback DataFrame if failed/empty.

fn read_csv_or(path: String, fallback: momijo.dataframe.DataFrame) -> momijo.dataframe.DataFrame
    var df = momijo.dataframe.read_csv(path)
    if df.width() == 0 or df.height() == 0:
        return fallback
    return df

# Try writing CSV; simply forward the boolean status.

fn try_write_csv(df: momijo.dataframe.DataFrame, path: String) -> Bool
    return momijo.dataframe.write_csv(df, path)


# ---- Auto-merged additions (from fn_final_english) ----
# All comments are English-only. Duplicated names/signatures were skipped.

    fn __init__(out out self, outout self, labels: List[String]): self.labels = labels

    fn __init__(out out self, outout self, level1: List[String], level2: List[String]):
        self.level1 = level1; self.level2 = level2

    fn __init__(out out self, outout self, categories: List[String], codes: List[Int]):
        self.categories = categories; self.codes = codes

    fn __init__(out out self, outout self, seed: UInt64): self.state = seed

    fn __init__(out out self, outout self, display_rows: Int): self.display_rows = display_rows

    fn __init__(out out self, outout self, left: Float64, right: Float64, closed_left: Bool = True):
        self.left = left; self.right = right; self.closed_left = closed_left

    fn __init__(out out self, outout self,s: UInt64):
        self.state =s

    fn next_f64(mut self) -> Float64
        return Float64(self.next_u32()) / 4294967296.0

rlier pattern) ------------

fn max(xs: List

st[Bool], idx: List[Int]) -> List[Bo

fn take(xs: List

xs: List[Int64], idx: List[Int]) -> List[I

-> List[String]
    return take_str(xs, idx)

struct SimpleIndex:
    var labels: List[String]

struct Categorical:
    var categories: List[String]
    var codes: List[Int]

struct RNG:
    var state: UInt64

struct FileHandle:

struct Interval:
    var left: Float64; var right: Float64; var closed_left: Bool

fn str_eq(a: String, b: String) -> Bool return a == b

fn contains_string(xs: List[String], x: String) -> Bool
    var i = 0
    while i < len(xs):
        if xs[i] == x: return True
        i += 1
    return False

fn unique_strings_list(xs: List[String]) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(xs):
        if not contains_string(out, xs[i]):
            out.append(xs[i])
        i += 1
    return out

fn min_f64(xs: List[Float64

4]) -> Float64
    if len(xs) == 0: return 0.0
    var m = xs[0]; var i = 1
    while i < len(xs):
        if xs[i] > m: m = xs[i]

1; i = 1
    var acc: Int64 = 0
    while i < len(s):
        var ch = s[i]
        if ch >= 48 and ch <= 57:
            acc = acc * 10 + Int64(ch - 48)
        else:
            break
        i += 1
    return sign * acc

fn find_col(df: DataFrame, name: String) -> Int
    var i = 0
    while i < df.width():
        if df.names[i] == name: return i
        i += 1
    return -1
# Textual, dependency-free preview

fn section_02_orders() -> DataFrame
    var ts = ["2025-01-01 09:00:00","2025-01-01 09:05:00","2025-01-01 09:10:00","2025-01-01 09:10:00"]  # dup at 09:10
    var order_id = [101,102,103,103]  # duplicate id on purpose
    var qty = [1,2,0,3]               # a zero and missing-like scenario
    var names = ["ts","order_id","qty"]
    var cols = [col_str("ts", ts), col_i64("order_id", order_id), col_i64("qty", qty)]
    var df = df_make(names, cols)
    preview(df, String("2) time-indexed orders (simulated)"))
    return df
# ----------------------------------------------------------------------------
# 3) Casting / convert_dtypes / infer_objects
# ----------------------------------------------------------------------------

fn section_03_cast(df: DataFrame) -> DataFrame
    # Convert 'vip' from string to 0/1 (demo for convert_dtypes)
    var idx = find_col(df, String("vip"))
    if idx >= 0:
        var vals = List[Int64]()
        var r = 0
        while r < df.height():
            var s = df.cols[idx].get_string(r)
            vals.append(Int64(1) if s == String("True") else Int64(0))
            r += 1
        df.cols[idx] = col_i64(String("vip"), vals)
    preview(df, String("3) cast/convert"))
    return df
# ----------------------------------------------------------------------------
# 4) Index & MultiIndex basics (placeholder structures)
# ----------------------------------------------------------------------------

fn section_04_index_basics(df: DataFrame)
    var idx = SimpleIndex(["r0","r1","r2","r3","r4"])
    var mix = SimpleMultiIndex(["A","A","B","B","B"], ["x","y","x","x","y"])
    print(String("4) Index basics: ")+String(len(idx.labels))+String(" labels; MultiIndex levels=")+String(len(mix.level1)))
# ----------------------------------------------------------------------------
# 5) Index ops: union/intersection/difference/symmetric_difference
# ----------------------------------------------------------------------------

fn list_union(a: List[String], b: List[String]) -> List[String]
    var out = a
    var i = 0
    while i < len(b):
        if not contains_string(out, b[i]): out.append(b[i])
        i += 1
    return out

fn list_intersection(a: List[String], b: List[String]) -> List[String]
    var out = List[String](); var i = 0
    while i < len(a):
        if contains_string(b, a[i]): out.append(a[i])
        i += 1
    return out

fn list_difference(a: List[String], b: List[String]) -> List[String]
    var out = List[String](); var i = 0
    while i < len(a):
        if not contains_string(b, a[i]): out.append(a[i])
        i += 1
    return out

fn list_symdiff(a: List[String], b: List[String]) -> List[String]
    var u = list_union(a,b); var ii = list_intersection(a,b)
    return list_difference(u, ii)

fn section_05_index_ops()
    var a = ["r0","r1","r2"]; var b = ["r2","r3"]
    print(String("5) union=")+String(len(list_union(a,b)))+String(", inter=")+String(len(list_intersection(a,b))))
# ----------------------------------------------------------------------------
# 6) Selection: loc/iloc/at/iat + slicers + IndexSlice (simplified)
# ----------------------------------------------------------------------------

fn iloc_row(df: DataFrame, r: Int) -> List[Stri

String]
    var out = List[String](); var r = 0
    while r < df.height(): out.append(df.cols[cidx].get_string(r)); r += 1
    r

)+String(len(c0)))
# ----------------------------------------------------------------------------
# 7) Boolean masks / where / mask / between / isin
# ----------------------------------------------------------------------------

fn between_i64(x: Int64, a: Int64, b: Int64) -> Bool return x >= a and x <= b

fn isin_string(s: String, universe: List[String]) -> Bool return contains_string(universe, s)

fn isna_str(s: String) -> Bool return s == String("") or s == String("NA") or s == String("NaN")

fn value_counts_str(xs: List[String]

t64](); var i = 0
    while i

if xs[j] == k: cnt += 1
            j += 1
        counts.append(cnt); i += 1
    return df_make(["value","count"], [col_str("value", keys), col_

var j = 0
        while j < len(old) and (i+j) < len(s):
            if s[i+j] not = old[j]: match = False
            j += 1
        if match and len(old) > 0 and i + len(old) <= len(s):
            out = out + newv; i = i + len(old)
        else:
            out = out + String(s[i]); i += 1
    return out
# ----------------------------------------------------------------------------
# 11) Categoricals: reorder/remove_unused_categories (demo)
# ----------------------------------------------------------------------------

    fn remove_unused(out self)

self.categories = new_order
# ----------------------------------------------------------------------------
# 12) Build fact via merge/join variants (left join demo)
# ----------------------------------------------------------------------------

fn argsort_f64(xs: List[Float64], asc: Bool = True) -> List[Int]
    # simple selection sort indices
    var idxs = List[Int](); var i = 0
    while i < len(xs): idxs.append(i); i += 1
    var a = 0
    while a < len(xs):
        var b = a + 1
        while b < len(xs):
            var cond = xs[idxs[b]] < xs[idxs[a]] if asc else xs[idxs[b]] > xs[idxs[a]]
            if cond:
                var tmp = idxs[a]; idxs[a] = idxs[b]; idxs[b] = tmp
            b += 1
        a += 1
    return idxs

fn rank_dense_f64(xs: List[Float64]) -> List[Int]
    var order = argsort_f64(xs, True)
    var ranks = List[Int](len(xs), 0)
    var rnk = 1; var i = 0
    while i < len(order):
        if i == 0 or xs[order[i]] not = xs[order[i-1]]:
            rnk = rnk if i == 0 else rnk + 1
        ranks[order[i]] = rnk; i += 1
    return ranks
# ----------------------------------------------------------------------------
# 16) GroupBy: agg (dict & named), transform, apply, filter, observed (demo)
# ----------------------------------------------------------------------------

fn tz_localize_utc(ts: List[String]) -> List[String]
    var out = List[String](); var i = 0
    while i < len(ts): out.append(ts[i] + String("+00:00")); i += 1
    return out

fn tz_convert(ts: List[String], target: String) -> List[String]
    var out = List[String](); var i = 0
    while i < len(ts): out.append(ts[i] + String("->") + target); i += 1
    return out
# ----------------------------------------------------------------------------
# 20) Query / eval performance helpers (illustrative)
# ----------------------------------------------------------------------------
# Pass a simple predicate over a row (as co

_once(s: String, sep: String) -> (String, String)
    var i = 0
    while i + len(sep) <= len(s):
        var ok = True; var j = 0
        while j < len(sep):
            if s[i+j] not = sep[j]: ok = False
            j += 1
        if ok:
            return (s.slice(0,i), s.slice(i+len(sep), len(s)))
        i += 1
    return (s, String(""))

fn str_strip(s: String) -> String
    var l = 0; var r = len(s)
    while l < r and (s.bytes()[l] == UInt8(32) or s.bytes()[l] == UInt8(9)): l += 1
    while r > l and (s.

atetime ops: floor/ceil/round/normalize/to_period (placeholders)
# ------------------------------------------------------

= ts[i]; var cut = 10 if len(t) >= 10 else len(t)

]
    var n = len(xs); var out = List[Float64](); var i = 0
    while i < n:
        var s = 0.0; var cnt = 0; var k = i - win + 1
        if k < 0: k = 0
        while k <= i:
            s += xs[k]; cnt += 1; k += 1
        out.append(s/Float64(cnt)); i += 1
    return out
# ----------------------------------------------------------------------------
# 24) Correlation/Covariance,

t

fn corr_f64(x: List[Float64], y: List[Float64]) -> Float64
    var n = len(x); if n == 0 or len(y) not = n: return 0.0
    var sx=0.0; var sy=0.0; var i=0
    while i<n: sx+=x[i]; sy+=y[i]; i+=1
    var mx = sx/Float64(n); var

ustom percentiles, memory_usage (demo)
# ----------------------------------------------------------------------------

fn percentile_f64(xs: List[Float64], p: Float64) -> Float64
    if len(xs) == 0: return 0.0
    var idx = Int(Float64(len(xs)-1) * p)
    return xs[argsort_f64(xs, True)[idx]]
# ----

f_last_before(xs: List[Float64], t_index: List[Int], t: Int) -> Float64
    var best_t = -2147483648; var best_v = 0.0; var i = 0
    while i < len(xs):
        if t_index[i] <= t and t_index[i] > best_t:
            best_t = t_index[i]; bes

fn close(self) pass

fn open_file_for_write(path: String) -> FileHandle return FileHandle()

fn open_file_for_read(path: String) -> FileHandle return FileHandle()

fn write_json_min(df: DataFrame, path: String) pass

fn write_pickle_min(df: DataFrame, path: String) pass

fn with_options(mut opt: Options, action: String)
    # no-op context
    print(String("with_options: display_rows=") + String(opt.display_rows) + String(" -> ") + action)
# ----------------------------------------------------------------------------
# 30) Final small checks
# ----------------------------------------------------------------------------

fn m

-----------------

    fn contains(self, x: Float64) -> Bool
        return (x >= self.left and x < self.right) if self.closed_left else (x > self.left and x <= self.right)

fn searchsorted_f64(sorted: List[Float64], x: Float64) -> Int
    var i = 0
    while i < len(sorted) and sorted[i] < x: i += 1
    return i
# ----------------------------------------------------------------------------
# 32) GroupBy on levels & GroupBy+Resample (placeholders)
# ------------------------------------------------

0
    while i < len(series):
        var s = 0.0; var k = 0
        while k < freq and i + k < len(series): s += series[i+k

:
        var s = 0.0; var cnt = 0; var k = i - win + 1
        if k < 0: k = 0
        while k <= i:
            s += (xs[k] if xs.bytes()[k] >= UInt8(0).0 else -xs[k]); cnt += 1; k += 1
        out.append(s/Float64(cnt)); i += 1
    return out
# ----------------------------------------------------------------------------
# 34) Expanding.corr with shifted series; EWM span (placeholders)
# ----------------------------------------------------------------------------

fn expanding_corr_last(x: List[Float64], y: List[Float64]) -> List[Float64]
    var out = List[Float64](); var i = 1
    while i <= len(x):
        out.append(corr_f64(x.slice(0,i), y.slice(0,i))); i += 1
    return out
# ----------------------------------------------------------------------------
# 35) merge extras: indicator, validate (demo flags)
# ----------------------------------------------------------------------------

fn merge_indicator_validate(a: DataFrame, b: DataFrame, on_a: String, on_b: String, validate: String) -> DataFrame
    # call left_join and append an indicator column
    var tmp = left_join(a, b, on_a, on_b)
    var src = List[String](); var i = 0
    while i < tmp.height(): src.append(String("left_only_or_both")); i += 1
    var names = tmp.names; var cols = tmp

assign-chain deps, sort_values key (illustrative)
# ----------------------------------------------------------------------------

fn sort_values_key(xs: List[String]) -> List[Int]
    # sort by length of string (example key)
    var idxs = List[Int](); var i = 0
    while i < len(xs): idxs.append(i); i += 1
    var a = 0
    while a < len(xs):
        var b = a + 1
        while b < len(xs):
            if len(xs[idxs[b]]) < len(xs[idxs[a]]):
                var t = idxs[a]; idxs[a] = idxs[b]; idxs[b] = t
            b += 1
        a += 1
    return idxs
# ----------------------------------------------------------------------------
# 37) get_dummies options, explode multi, value_counts bins
# ----------------------------------------------------------------------------

fn value_counts_bins(xs:

f64(xs); var w = (mx - mn)/Float64(bins)
    var counts = List[Int64](bins, 0); var i = 0
    while i < len(xs):
        var b = Int((xs[i]-mn)/w) if w not = 0.0 else 0
        if b < 0: b = 0
        if b >= bins: b = bins - 1
        counts[b] = counts[b] + 1; i += 1
    var labels = List[String](); var j = 0
    while j < bins:
        var l = mn + Float64(j)*w; var r = l + w
        labels.append(String("[")+String(l)+String(",")+String(r)+String(")")); j += 1
    return df_make(["bin","count"], [col_str("bin", labels), col_i64("count", counts)])
# ----------------------------------------------------------------------------
# 38) Index/rename_ax

ype(errors='ignore'), drop_duplicates, sample replace, to_dict/json_normalize
# ----------------------------------------------------------------------------

fn

t()) + String(", cols=") + String(df.width()))
    # Accessing public field `names` for a quick header dump
    print(String("columns: ") + join_names(df.names))

fn pad2(x: Int) -> String
    var s = String(x)
    if x < 10:
        s = String("0") + s
    return s

fn gen_dates_12h_from_2025_01_01(

ut



# ---------------- Utils ----------------

fn print_dtypes(df: DataFrame)
    print(String("dtypes:"))
    var i = 0
    while i < len(df.names):
        print(df.names[i] + String(": ") + dtype_name_of_column(df.cols[i]))
        i += 1




# Simple argsort for Int64 list

fn argsort_i64(xs: List[Int64]) -> List[Int]
    var idx = List[Int]()
    var i = 0
    while i < len(xs):
        idx.append(i)
        i += 1
    # insertion sort by xs[idx]
    var j = 1
    while j < len(idx):
        var ke

(idx):
        var key = idx[j]
        var k = j - 1
        while k >= 0 and (a[idx[k]] > a[key] or (a[idx[k]] == a[key] and b[idx[k]] > b[key])):
            idx[k+1] = idx[k]
            k -= 1
        idx[k+1] = key
        j += 1
    return idx

# Take rows for parallel lists

fn take_i64(xs: List[Int64], idx: List[Int]) -> List[Int64]
    var out = List[Int64]()
    var i = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_f64(xs: List[Float64], idx: List[Int]) -> List[Float64]
    var out = List[Float64]()
    var i = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_str(xs: List[String], idx: List[Int]) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

fn take_bool(xs: List[Bool], idx: List[Int]) -> List[Bool]
    var out = List[Bool]()
    var i = 0
    while i < len(idx):
        out.append(xs[idx[i]])
        i += 1
    return out

# Set-like ops on String lists (unique-ify then operate)

fn unique_str(xs: List[String]) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(xs):
        var seen = False
        var j = 0
        while j < len(out):
            if out[j] == xs[i]:
                seen = True
                break
            j += 1
        if not seen:
            out.append(xs[i])
        i += 1
    return out

fn union_str(a: List[String], b: List[String]) -> List[String]
    var out = unique_str(a)
    var i = 0
    while i < len(b):
        var seen = False
        var j = 0
        while j < len(out):
            if out[j] == b[i]:
                seen = True
                bre

ut.append(a[i])
        i += 1
    return out

fn symdiff_str(a: List[String], b: List[String]) -> List[String]
    var ab = diff_str(a, b)
    var ba = diff_str(b, a)
    return union_str(ab, ba)

fn print_i64_head(label: String, xs: List[Int64], k: Int)
    var n = len(xs)
    var limit = k
    if limit > n:
        limit = n

    var line = label + String(": [")
    var i = 0
    while i < limit:
        line = line + String(xs[i])
        if i + 1 < limit:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)


# Print first K items of a List[Float64], plus its length.

fn print_f64_head(label: String, xs: List[Float64], k: Int)
    var n = len(xs)
    var limit = k
    if limit > n:
        limit = n

    var line = label + String(": [")
    var i = 0
    while i < limit:
        line = line + String(xs[i])
        if i + 1 < limit:
            line = line + String(", ")
        i += 1
    line = line + String("]  (len=") + String(n) + String(")")
    print(line)


# Print first K items of a List[String], plus its length.

    fn rep(s: String, n: Int) -> String
        var out = String(""); var i = 0
        while i < n: out = out + s; i += 1
        return out

    fn pad(s: String, w: Int) -> String
        var out = s; var i = len(out)
        while i < w: out = out + String(" "); i += 1
        return out

    # NOTE: [Translated] Comment text was originally in Persian; please clarify meaning.
    var col_w = 14
    var left_w = 6
    var ncols = df.width()

y in Persian; please clarify meaning.
    var start = 0
    var count = rows
    if mode == String("all"):
        count = nrows; start = 0
    else:
        if count > nrows: count = nrows
        if mode == String("tail"):
            start = nrows - co

rows) + String(", ") + String(ncols) + String(")"))

    var names = String(""); var c = 0
    while c < ncols:
        names = names + df.names[c]
        if c + 1 < ncols: names = names + String(", ")
        c += 1
    print(String("columns: ") + names)

    # NOTE: [Translated] Comment text was originally in Persian; please clarify meaning.
    var top = String("+") + rep(String("-"), left_w)
    var mid = String("+") + rep(String("="), left_w)
    var i = 0
    while i < ncols:
        top = top + String("+") + rep(String("-"), col_w)
        mid = mid + String("+") + rep(String("="), col_w)
        i += 1
    top = top + String("+"); mid = mid + String("+")
    print(top)

    # NOTE: [Translated] Comment text was originally in Persian; please clarify meaning.
    var hdr = String("|") + pad(String("#"), left_w)
    i = 0
    while i < ncols:
        hdr = hdr + String("|") + pad(df.names[i], col_w)
        i += 1
    hdr = hdr + String("|")
    print(hdr)
    print(mid)

    # NOTE: [Translated] Comment text was originally in Persian; please clarify meaning.
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


# Return cell as String (pick ONE line that compiles for your API)

fn intersect_str(a: List[String], b: List[String]) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(a):
        var j = 0
        while j < len(b):
            if a[i] == b[j]:
                var seen = False
                var k = 0
                while k < len(out):
                    if out[k] == a[i]: seen = True; break
                    k += 1
                if not seen: out.append(a[i])

x: Float64) -> Float64
    if x <= 0.0: return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g

# -------------- Build sample frames --------------

fn is_alpha_code(c: UInt8) -> Bool
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

fn is_digit_code(c: UInt8) -> Bool
    return c >= 48 and c <= 57

fn rpad(s: String, width: Int, fill: String) -> String
    var out = s

[String]
    # Only February 2025; naive calendar for demo
    var out = List[String]()
    var day = start_day
    var hour = start_hour
    v

s)
        var total = hour * 60 + minute + step_min
        hour = (total // 60) % 24
        minute = total % 60
        if total >= 24 * 60:

xt_u32(mut self) -> UInt32
        self.state = self.state * 6364136223846793005 + 1
        return UInt32((self.state >> 32) &  UInt8(0xFFFFFFFF))

fn

Int8]()
    var i = 0
    while i < len(s):
        out.append(UInt8(ord(s[i]) &  UInt8(0xFF)))
        i += 1
    return out

fn bytes_to_string(bs: List[UInt8]) -> String
    var out = String("")
    var i = 0
    while i < len(bs):
        out = out + String(Char(Int(bs[i])))
        i += 1
    return out

# -------------------------------------------------
# Little-endian helpers
# -------------------------------------------------

fn u32_to_le_bytes(x: UInt32) -> List[UInt8]
    var b = List[UInt8]()
    b.append(UInt8(x &  UInt8(0xFF)))
    b.append(UInt8((x >> 8) &  UInt8(0xFF)))
    b.append(UInt8((x >> 16) &  UInt8(0xFF)))
    b.append(UInt8((x >> 24) &  UInt8(0xFF)))
    return b

fn u64_to_le_bytes(x: UInt64) -> List[UInt8]
    var b = List[UInt8]()
    var i = 0
    while i < 8:
        b.append(UInt8((x >> UInt64(8 * i)) & UInt64(0xFF)))
        i += 1
    return b

fn i64_to_le_bytes(x: Int64) -> List[UInt8]
    return u64_to_le_bytes(UInt64(x))

# -------------------------------------------------
# File I/O (bytes)
# -------------------------------------------------

fn read_bytes(path: String) -> List[UInt8]
    try:
        var p = pathlib.Path(path)
        var data = p.read_bytes()
        return data
    except e:
        return List[UInt8]()

fn write_bytes(path: String, data: List[UInt8]) -> Bool
    try:
        var p = pathlib.Path(path)
        p.write_bytes(data)
        return True
    except e:
        return False

# -------------------------------------------------
# Minimal JSON header builder (no parser dependency)
# -------------------------------------------------

fn json_escape(s: String) -> String
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == '"':
            out = out + String("\\\"")
        elif ch == '\\':
            out = out + String("\\\\")
        elif ch == '\n':
            out = out + String("\\n")
        elif ch == '\r':
            out = out + String("\\r")
        elif ch == '\t':
            out = out + String("\\t")
        else:
            out = out + String(ch)
        i += 1
    return out

fn build_header_json(df: DataFrame, floats_ascii: Bool = True) -> String
    var s = String("{")
    s = s + String("\"format\":\"MNP1\",\"floats_ascii\":") + (String("true") if floats_ascii else String("false"))
    s = s + String(",\"nrows\":") + String(df.height())
    s = s + String(",\"ncols\":") + String(df.width())
    s = s + String(",\"cols\":[")
    var i = 0
    while i < df.width():
        var name_esc = json_escape(df.names[i])
        var col = df.get_column_at(i)
        var ty = String("str")
        if col.is_bool():
            ty = String("bool")
        elif col.is_i64():
            ty = String("i64")
        elif col.is_f64():
            ty = String("f64")
        s = s + String("{\"name\":\"") + name_esc + String("\",\"type\":\"") + ty + String("\"}")
        if i + 1 < df.width():
            s = s + String(",")
        i += 1
    s = s + String("]}")
    return s

# -------------------------------------------------
# Public API: write_pickle/read_pickle (MNP v1)
# -------------------------------------------------

fn write_pickle(df: DataFrame, path: String) -> Bool
    var floats_ascii = True
    var header = build_header_json(df, floats_ascii)

    var out = List[UInt8]()

    # MAGIC "MNP1PKL\0"
    out.append(UInt8(0x4D))
    out.append(UInt8(0x4E))
    out.append(UInt8(0x50))
    out.append(UInt8(0x31))
    out.append(UInt8(0x50))
    out.append(UInt8(0x4B))
    out.append(UInt8(0x4C))
    out.append(UInt8(0x00))

    # header length (u64 LE)
    var hbytes = str_to_bytes(header)
    var hlen = UInt64(len(hbytes))
    var hlen_le = u64_to_le_bytes(hlen)
    var i = 0
    while i < len(hlen_le):
        out.append(hlen_le[i])
        i += 1

    # header bytes
    i = 0
    while i < len(hbytes):
        out.append(hbytes[i])
        i += 1

    # column blobs
    var nrows = df.height()
    var ncols = df.width()

    var c = 0
    while c < ncols:
        var col = df.get_column_at(c)

        if col.is_bool():
            var r = 0
            while r < nrows:
                var b = col.get_bool(r)
                if b:
                    out.append(UInt8(1))
                else:
                    out.append(UInt8(0))
                r += 1

        elif col.is_i64():
            var r2 = 0
            while r2 < nrows:
                var v = col.get_i64(r2)
                var le = i64_to_le_bytes(v)
                var k = 0
                while k < len(le):
                    out.append(le[k])
                    k += 1
                r2 += 1

        elif col.is_f64():
            if floats_ascii:
                var lc = u32_to_le_bytes(UInt32(nrows))
                var kk = 0
                while kk < len(lc):
                    out.append(lc[kk])
                    kk += 1
                var r3 = 0
                while r3 < nrows:
                    var s = String(col.get_f64(r3))
                    var bytes = str_to_bytes(s)
                    var ln = u32_to_le_bytes(UInt32(len(bytes)))
                    var jj = 0
                    while jj < len(ln):
                        out.append(ln[jj])
                        jj += 1
                    var ii = 0
                    while ii < len(bytes):
                        out.append(bytes[ii])
                        ii += 1
                    r3 += 1
            else:
                return False

        else:
            var utf8 = List[UInt8]()
            var offsets = List[UInt32]()
            offsets.append(UInt32(0))
            var r4 = 0
            while r4 < nrows:
                var s = col.value()_to_string(r4)

offsets.append(UInt32(len(utf8)))
                r4 += 1

            var cnt = u32_to_le_bytes(UInt32(len(offsets)))
            var q = 0
            while q < len(cnt):
                out.append(cnt[q])
                q += 1

            var idx = 0
            while idx < len(offsets):
                var o_le = u32_to_le_bytes(offsets[idx])
                var m = 0
                while m < len(o_le):
                    out.append(o_le[m])
                    m += 1
                idx += 1

            var z = 0
            while z < len(utf8):
                out.append(utf8[z])
                z += 1

        c += 1

    return write_bytes(path, out)

fn read_pickle(path: String) -> DataFrame
    var data = read_bytes(path)
    if len(data) < 16:
        return DataFrame()

    if not (
        data[0] == UInt8(0x4D) and data[1] == UInt8(0x4E) and data[2] == UInt8(0x50) and data[3] == UInt8(0x31) and
        data[4] == UInt8(0x50) and data[5] == UInt8(0x4B) and data[6] == UInt8(0x4C) and data[7] == UInt8(0x00)
    ):
        return DataFrame()

    var hlen: UInt64 = 0
    var j = 0
    while j < 8:
        hlen = hlen | (UInt64(data[8 + j]) << UInt64(8 * j))
        j += 1

    var pos = 16
    if pos + Int(hlen) > len(data):
        return DataFrame()

    var header_bytes = List[UInt8]()
    var i = 0
    while i < Int(hlen):
        header_bytes.append(data[pos + i])
        i += 1
    pos += Int(hlen)
    var header = bytes_to_string(header_bytes)

    fn find_int(h: String, key: String, default_val: Int) -> Int
        var p = h.find(key + String("\":"))
        if p < 0:
            return default_val
        var q = p + len(key) + 2
        var val = 0
        var neg = False
        while q < len(h) and not ((ord(h[q]) >= 48 and ord(h[q]) <= 57) or h[q] == '-'):
            q += 1
        if q < len(h) and h[q] == '-':
            neg = True
            q += 1
        while q < len(h) and (ord(h[q]) >= 48 and ord(h[q]) <= 57):
            val = val * 10 + Int(ord(h[q]) - 48)
            q += 1
        if neg:
            return -val
        return val

    fn find_bool(h: String, key: String, default_val: Bool) -> Bool
        var p = h.find(key + String("\":"))
        if p < 0:
            return default_val
        var q = p + len(key) + 2
        while q < len(h) and (h[q] == ' ' or h[q] == '\n' or h[q] == '\t'):
            q += 1
        if q + 3 < len(h) and h[q] == 't':
            return True
        if q + 4 < len(h) and h[q] == 'f':
            return False
        return default_val

    fn find_next_quote(h: String, start: Int) -> Int
        var p = start
        while p < len(h):
            if ord(h[p]) == 34:
                return p
            p += 1
        return -1

# ---- Unified overload wrappers (auto-generated) ----

fn between(x: Int64, a: Int64, b: Int64) -> Bool
    return between_i64(x, a, b)

fn isna(s: String) -> Bool
    return isna_str(s)

fn value_counts(xs: List[String]) -> DataFrame
    return value_counts_str(xs)

fn argsort(xs: List[Float64], asc: Bool = True) -> List[Int]
    return argsort_f64(xs, asc)

fn argsort(xs: List[Float64]) -> List[Int]
    return argsort_f64(xs)

fn argsort(xs: List[Int64]) -> List[Int]
    return

dense_f64(xs)

fn legacy_cumsum(xs: List[Int64]) -> List[Int64]
    return cumsum_i64(xs)

fn corr(x: List[Float64], y: List[Float64]) -> Float64
    return corr_f64(x, y)

fn percentile(xs: List[Float64], p: Float64) -> Float64
    return percentile_f64(xs, p)

fn searchsorted(sorted: List[Float64], x: Float64) -> Int
    return searchsorted_f64(sorted, x)

fn unique(xs: List[String]) -> List[String]
    return unique_str(xs)

fn union(a: List[String], b: List[String]) -> List[String]
    return union_str(a, b)

fn diff(a: List[String], b: List[String]) -> List[String]
    return diff_str(a, b)

fn symdiff(a: List[String], b: List[String]) -> List[String]
    return symdiff_str(a, b)

fn intersect(a: List[String], b: List[String]) -> List[String]
    return intersect_str(a, b)

fn sqrt(x: Float64) -> Float64
    return sqrt_f64(x)

fn str_to(s: String) -> List[UInt8]
    return str_to_bytes(s)

fn u32_to_le(x: UInt32) -> List[UInt8]
    return u32_to_le_bytes(x)

fn u64_to_le(x: UInt64) -> List[UInt8]
    return u64_to_le_bytes(x)

fn i64_to_le(x: Int64) -> List[UInt8]
    return i64_to_le_bytes(x)

fn read(path: String) -> List[UInt8]
    return read_bytes(path)


# Pretty print a DataFrame using real Column getters (no external deps).
# Safe to call anywhere. Uses Column.get_string for display.
fn preview(df: DataFrame, title: String = String("DF")) -> None
    print(String("\n--- ") + title + String(" ---"))
    print(String("shape: (") + String(df.height()) + String(", ") + String(df.width()) + String(")"))
    # header
    var i = 0
    var header = String("")
    while i < df.width():
        header = header + df.names[i]
        if i + 1 < df.width(): header += ", "
        i += 1
    print(header)
    # rows (max 5)
    var rows = if df.height() < 5 { df.height() } else { 5 }
    var r = 0
    while r < rows:
        var line = String("")
        var c = 0
        while c < df.width():
            line = line + df.cols[c].get_string(r)
            if c + 1 < df.width(): line += ", "
            c += 1
        print(line)
        r += 1

# Very simple Float64 parser (digits and one dot); returns 0.0 on failure.
fn parse_f64_or_zero(s: String) -> Float64
    var has_dot = False
    var i = 0
    var sign: Float64 = 1.0
    if len(s.bytes()) > 0 and s.bytes()[0] == UInt8(45):  # '-'
        sign = -1.0
        i = 1
    var int_part: Int64 = 0
    var frac_part: Int64 = 0
    var frac_scale: Float64 = 1.0
    while i < len(s):
        var ch = s[i]
        if ch == 46 and not has_dot:  # '.'
            has_dot = True
        elif ch >= 48 and ch <= 57:
            var d = Int64(ch - 48)
            if not has_dot:
                int_part = int_part * 10 + d
            else:
                frac_part = frac_part * 10 + d
                frac_scale = frac_scale * 10.0
        else:
            break
        i += 1
    return sign * (Float64(int_part) + Float64(frac_part)/frac_scale)

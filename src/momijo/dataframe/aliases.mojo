# Project:      Momijo
# Module:       src.momijo.dataframe.aliases
# File:         aliases.mojo
# Path:         src/momijo/dataframe/aliases.mojo
#
# Description:  src.momijo.dataframe.aliases — focused Momijo functionality with a stable public API.
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
#   - Key functions: max, col, col, isna, combine_first, argsort, argsort, rank_dense ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array_stats import max
from momijo.dataframe.align import combine_first_str
from momijo.dataframe.api import col_i64
from momijo.dataframe.column import Column
from momijo.dataframe.column import col, combine_first, name
from momijo.dataframe.describe import percentile_f64
from momijo.dataframe.helpers import argsort, corr, i64_to_le, isna, rank_dense, str_to, symdiff, take, u64_to_le, union
from momijo.dataframe.helpers import between_i64, list_difference as _diff_impl, unique_strings_list as _unique_impl
from momijo.dataframe.interval import searchsorted_f64
from momijo.dataframe.io_bytes import u64_to_le_bytes
from momijo.dataframe.io_files import read_bytes, write_bytes
from momijo.dataframe.sorting import argsort_f64 as _argsort_f64, argsort_i64 as _argsort_i64
from momijo.dataframe.stats_core import cumsum_i64, sqrt_f64
assert(dataframe is not None, String("dataframe is None"))
from momijo.dataframe.value()_counts import value_counts_str as _value_counts_str
from momijo.extras.stubs import Li, Lis, Strin, _argsort_f64, _argsort_i64, _diff_impl, _unique_impl, builders, from, momijo, return, value

# MIT License
# Project: momijo.dataframe
# File: momijo/dataframe/aliases.mojo
#
# A thin facade of short, pandas-like helper names that forward to core implementations.
# Added carefully to avoid duplicating logic; safe to import everywhere.

# Min/Max wrappers

fn max(xs

builders (overloads)
fn col(name: Strin

g, vals: List[Float64]) -> Column retu

rn col_i64(name, vals)
fn col(name: String, vals: List[String]) -> Column  re

t64) -> Bool return between_i64(x, a, b)
fn isna(s: String) -> Bool return is

# Prefer dedicated module if present; else fallback to building tiny DF here
    return _value_

False, prefix: String = String

df, col, drop_first, prefix)

# Combine-first
fn combine_first(a: Column, b: Column) -> Column return combine_first_str(a, b)

# Sorting/ranking (overloads)
fn argsort(xs: List[Float64], asc: Bool = True) -> List[Int] return _argsort_f64(xs, asc)
fn argsort(xs: List[Int64]) -> List[Int] return _argsort_i64(xs)
fn rank_dense(xs: List[Float64]) -> List[Int] return r

return cumsum_i64(xs)
fn corr(x: List[Float64], y: List[Flo

oat64 return percentile_f64(xs, p)

# Searc

return searchsorted_f64(sorted, x)

# Take (overloads)
fn take(

fn take(xs: List[Float64], idx: Lis

4], idx: List[Int]

[Int]) -> List[String] return tak

rn _unique_impl(xs)
fn union(a: List[String], b: Li

) -> List[String] return _diff_impl(a, b)
fn symdiff(a: List[

t(a: List[String], b: List[String])

n sqrt_f64(x)

# Bytes/IO short names
fn str_to(s: String) -> List[UInt8] return str

u64_to_le(x: UInt64) -> List[UInt8] return u64_to_le_bytes(x)
fn i64_to_le(x: I

return read_bytes(path)
fn write(path: String, data: List[UInt8]) -> Bool return write_bytes(path, data
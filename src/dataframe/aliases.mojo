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
# File: src/momijo/dataframe/aliases.mojo

#   from momijo.core.error import describe
#   from momijo.core.errors import describe
# SUGGEST (alpha): from momijo.core.error import describe
# NOTE: Migrated: original 'fn cumsum' renamed to 'legacy_cumsum'; 'cumsum' now imported from stdlib
from momijo.extras.stubs import Copyright, Li, Lis, MIT, Migrated, SUGGEST, Strin, Take, _argsort_f64, _argsort_i64, _diff_impl, _unique_impl, builders, from, https, momijo, original, ranking, return, src, value
from momijo.dataframe.helpers import read
from momijo.dataframe.helpers import i64_to_le
from momijo.dataframe.helpers import u64_to_le
from momijo.dataframe.helpers import u32_to_le
from momijo.dataframe.helpers import str_to
from momijo.dataframe.helpers import sqrt
from momijo.dataframe.helpers import intersect
from momijo.dataframe.helpers import symdiff
from momijo.dataframe.helpers import diff
from momijo.dataframe.helpers import union
from momijo.dataframe.helpers import unique
from momijo.dataframe.helpers import take
from momijo.dataframe.helpers import searchsorted
from momijo.dataframe.helpers import percentile
from momijo.dataframe.helpers import corr
from momijo.dataframe.helpers import legacy_cumsum
from momijo.dataframe.helpers import rank_dense
from momijo.dataframe.helpers import argsort
from momijo.dataframe.column import combine_first
from momijo.dataframe.helpers import isna
from momijo.dataframe.helpers import between
from momijo.dataframe.column import col
from momijo.arrow_core.array_stats import max
from momijo.arrow_core.array_stats import min
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import name  # chosen by proximity
#   from momijo.arrow_core.arrays.boolean_array import value
#   from momijo.arrow_core.arrays.primitive_array import value
#   from momijo.arrow_core.arrays.string_array import value
#   from momijo.core.types import value
# SUGGEST (alpha): from momijo.arrow_core.arrays.boolean_array import value
#   from momijo.dataframe.helpers import value_counts_str
#   from momijo.dataframe.stats_transform import value_counts_str
#   from momijo.dataframe.value()_counts import value_counts_str
# SUGGEST (alpha): from momijo.dataframe.helpers import value_counts_str
from momijo.dataframe.value()_counts import value_counts_str as _value_counts_str
# MIT License
# Project: momijo.dataframe
# File: momijo/dataframe/aliases.mojo
#
# A thin facade of short, pandas-like helper names that forward to core implementations.
# Added carefully to avoid duplicating logic; safe to import everywhere.
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.api import col_bool, col_f64, col_i64, col_str
from momijo.dataframe.helpers import min_f64, max_f64, between_i64, isna_str
from momijo.dataframe.stats_core import cumsum_i64, corr_f64, sqrt_f64
from momijo.dataframe.describe import percentile_f64
from momijo.dataframe.interval import searchsorted_f64
from momijo.dataframe.encoding import get_dummies_str
from momijo.dataframe.take import take_bool, take_f64, take_i64, take_str
from momijo.dataframe.helpers import unique_strings_list as _unique_impl
from momijo.dataframe.helpers import list_union as _union_impl, list_difference as _diff_impl, list_symdiff as _symdiff_impl, list_intersection as _inter_impl
from momijo.dataframe.string_ops import compare_str_eq
from momijo.dataframe.sorting import argsort_f64 as _argsort_f64, argsort_i64 as _argsort_i64, rank_dense_f64
from momijo.dataframe.encoding import value_counts_bins as _value_counts_bins if exists
from momijo.dataframe.align import combine_first_str
from momijo.dataframe.io_bytes import str_to_bytes, u32_to_le_bytes, u64_to_le_bytes, i64_to_le_bytes
from momijo.dataframe.io_files import read_bytes, write_bytes

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


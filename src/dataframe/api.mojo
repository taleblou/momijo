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
# File: src/momijo/dataframe/api.mojo

from momijo.extras.stubs import Copyright, Helpers, MIT, SUGGEST, checks, column, for, from, https, if, import, len, logicals, momijo, names, raise, src, validate
from momijo.dataframe.column import col_bool
from momijo.dataframe.column import col
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import from_bool  # chosen by proximity
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
from momijo.dataframe.column import name  # chosen by proximity
from momijo.dataframe.expr import ConstVal
from momijo.core.errors import Error
from momijo.dataframe.expr import Expr
from momijo.dataframe.expr import Pred
from momijo.dataframe.column import col_f64
from momijo.dataframe.column import col_i64
from momijo.dataframe.column import col_str
from momijo.dataframe.generic_ops import eq
from momijo.dataframe.logical_plan import join
from momijo.dataframe.join import join_inner
from momijo.dataframe.join import join_left
from momijo.dataframe.join import join_right
from momijo.dataframe.helpers import preview_string
from momijo.dataframe.helpers import read_csv_or
from momijo.dataframe.logical_plan import sort
from momijo.dataframe.helpers import try_read_csv
from momijo.dataframe.helpers import try_write_csv
from momijo.dataframe.logical_plan import window
from momijo.dataframe.column import Column, ColumnTag
from momijo.dataframe.frame  import DataFrame

# If Bitmap is stable in your tree, uncomment the next line:
from momijo.dataframe.bitmap import Bitmap

# ---------------------------
# CSV I/O
# ---------------------------
from momijo.dataframe.io_csv import read_csv, write_csv

# ---------------------------
# Helpers (strict/lenient I/O + validation + preview)
# ---------------------------
from momijo.dataframe.helpers import (
    assert_nonempty,
    preview_string,
    read_csv_or_raise,
    write_csv_or_raise,
    try_read_csv,
    read_csv_or,
    try_write_csv
)

# ---------------------------
# (Optional) Logical plan + executor
# Uncomment when these modules compile cleanly.
# ---------------------------
from momijo.dataframe.logical_plan import (
    PLAN_SCAN, PLAN_FILTER, PLAN_PROJECT, PLAN_AGGREGATE,
    PLAN_SORT, PLAN_JOIN, PLAN_WINDOW
)
# from momijo.dataframe.exec import Executor

# ---------------------------
# (Optional) Joins
# Keep only existing, working functions.
# ---------------------------
from momijo.dataframe.join import (
    join_inner,
    join_left,
    join_right,
    join_outer,
    join_cross
)

# ---------------------------
# (Optional) Expressions / Optimizer
# Uncomment when available.
# ---------------------------
from momijo.dataframe.expr import (
    Expr, Pred, ConstVal,
    col, lit,
    # comparisons
    eq, ne, gt, ge, lt, le,
    # logicals (use and_/or_ if 'and'/'or' are reserved)
    and_, or_
)
from momijo.dataframe.optimizer import (
    Optimizer, optimize_logical_plan
)

# ---------------------------
# (Optional) Future growth
# ---------------------------
from momijo.dataframe.sort import sort_values
from momijo.dataframe.groupby import groupby, agg
from momijo.dataframe.window import window_fn


# ==== Convenience builders inserted by assistant ====
# Make a DataFrame from names + columns with validation.
fn df_make(names: List[String], cols: List[Column]) -> DataFrame
    var df = DataFrame(names, cols)
    # Optional: validate non-empty and consistent height
    # Comment out if you want to allow empty frames.
    return df

# Overloaded convenience: col(name, vals) for String values

# Overloaded convenience: col(name, vals) for Int64 values
fn col(name: String, vals: List[Int64]) ->

for Float64 values
fn col(name: String, vals: List[Float64]) -> Column
    return col_f64(name, vals)

# Strict maker: checks width/height co

ring], cols: List[Column]) -> DataFrame raises
    if len(names) not = len(cols):
        raise Error("df_make_strict: names/cols length mismatch")
    var h = 0
    if len(cols) > 0:
        h = cols[0].len()
    var i = 0
    while i < len(cols):
        if cols[i].len() not = h:
            raise Error("df_make_strict: column height mismatch at index " + String(i))
        i += 1
    return DataFrame(names, cols)

from momijo.dataframe.series_bool import SeriesBool

# Build a Bool column from values
fn col_bool(name: String, vals: List[Bool]) -> Column
    return Column.from_bool(SeriesBool(name, vals))


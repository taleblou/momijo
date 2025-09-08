# Project:      Momijo
# Module:       src.momijo.dataframe.api
# File:         api.mojo
# Path:         src/momijo/dataframe/api.mojo
#
# Description:  src.momijo.dataframe.api — focused Momijo functionality with a stable public API.
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
#   - Key functions: df_make, col, col, col_bool
#   - Error paths explicitly marked with 'raises'.
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.errors import Error
from momijo.dataframe.column import Column, col, col_bool, col_f64, from_bool, name
from momijo.dataframe.expr import (
from momijo.dataframe.expr import ConstVal, Expr, Pred
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.generic_ops import eq
from momijo.dataframe.helpers import (, preview_string, read_csv_or, try_read_csv, try_write_csv
from momijo.dataframe.join import (
from momijo.dataframe.join import join_inner, join_left, join_right
from momijo.dataframe.logical_plan import (
from momijo.dataframe.optimizer import (
from momijo.dataframe.series_bool import SeriesBool
from momijo.extras.stubs import column, for, if, len, names, raise

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
    PLAN_SCAN, PLAN_FILTER, PLAN_PROJECT, PLAN_AGGREGATE,
    PLAN_SORT, PLAN_JOIN, PLAN_WINDOW
)
# from momijo.dataframe.exec import Executor

# ---------------------------
# (Optional) Joins
# Keep only existing, working functions.
# ---------------------------
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
    Expr, Pred, ConstVal,
    col, lit,
    # comparisons
    eq, ne, gt, ge, lt, le,
    # logicals (use and_/or_ if 'and'/'or' are reserved)
    and_, or_
)
    Optimizer, optimize_logical_plan
)

# ---------------------------
# (Optional) Future growth
# ---------------------------

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

# Build a Bool column from values
fn col_bool(name: String, vals: List[Bool]) -> Column
    return Column.from_bool(SeriesBool(name, vals))
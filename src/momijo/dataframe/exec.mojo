# Project:      Momijo
# Module:       dataframe.exec
# File:         exec.mojo
# Path:         dataframe/exec.mojo
#
# Description:  dataframe.exec — Exec module for Momijo DataFrame.
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
#   - Structs: —
#   - Key functions: execute, exec_aggregate, groupby_multi, agg_on_col, agg_on_col_rows

from pathlib import Path
from pathlib.path import Path

from momijo.core.device import kind, unknown
from momijo.core.traits import one
from momijo.dataframe.categorical import reorder
from momijo.dataframe.column import Column, as_f64_or_nan, from_str
from momijo.dataframe.diagnostics import safe
from momijo.dataframe.utis import eval_expr, single
from momijo.dataframe.frame import DataFrame, filter_by_mask, get_column_index, select_columns, sort_by_multi
from momijo.dataframe.helpers import between, m, unique
from momijo.dataframe.join import join_anti, join_full, join_inner, join_left, join_right
from momijo.dataframe.logical_plan import aggregate, join, sort
from momijo.dataframe.frame import rolling_mean
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.window import cum_mean, cum_sum, dense_rank, lag, lead, rolling_mean, row_number
from momijo.ir.dialects.annotations import unit
from momijo.ir.passes.cse import find
from momijo.tensor.errors import OK, Unknown
from momijo.utils.logging import child
from momijo.utils.result import g
from momijo.visual.scene.facet import LogicalPlan, PLAN_AGGREGATE, PLAN_FILTER, PLAN_JOIN, PLAN_PROJECT, PLAN_SCAN, PLAN_SORT


fn execute(plan: LogicalPlan) -> DataFrame:
    if plan.kind == PLAN_SCAN:
# Return the already materialized DataFrame
        return plan.df

    elif plan.kind == PLAN_FILTER:
        var df = execute(plan.child)
# expr must evaluate to a boolean mask aligned with df rows
        var mask = eval_expr(df, plan.expr)
        return df.filter_by_mask(mask)

    elif plan.kind == PLAN_PROJECT:
        var df = execute(plan.child)
# Select/reorder a subset of columns
        return df.select_columns(plan.columns)

    elif plan.kind == PLAN_AGGREGATE:
        var df = execute(plan.child)
        return exec_aggregate(df, plan.group_keys, plan.aggs)

    elif plan.kind == PLAN_SORT:
        var df = execute(plan.child)
# Multi-key, multi-order sort
        return df.sort_by_multi(plan.sort_keys, plan.sort_asc)

    elif plan.kind == PLAN_JOIN:
# Evaluate both sides first (left-deep or bushy trees both OK)
        var left_df = execute(plan.child)
        var right_df = execute(plan.right)

# Stable string-based dispatch on join kind
        if plan.join_kind == String("inner"):
            return join_inner(left_df, right_df, plan.left_keys, plan.right_keys, plan.suffix_left, plan.suffix_right)
        elif plan.join_kind == String("left"):
            return join_left(left_df, right_df, plan.left_keys, plan.right_keys, plan.suffix_left, plan.suffix_right)
        elif plan.join_kind == String("right"):
            return join_right(left_df, right_df, plan.left_keys, plan.right_keys, plan.suffix_left, plan.suffix_right)
        elif plan.join_kind == String("full"):
            return join_full(left_df, right_df, plan.left_keys, plan.right_keys, plan.suffix_left, plan.suffix_right)
        else:
# Default to anti join for unknown kinds to avoid throwing.
            return join_anti(left_df, right_df, plan.left_keys, plan.right_keys)

    else:
# PLAN_WINDOW
        var df = execute(plan.child)

# Window/analytics functions (rolling + ranking + shifts + cumulatives)
        if plan.window_kind == String("rolling_mean"):
            return rolling_mean(
                df,
                plan.window_value_col,
                plan.window_param_int,
                plan.window_order_by,
                plan.window_partition_by
            )
        elif plan.window_kind == String("row_number"):
            return row_number(
                df,
                plan.window_order_by,
                plan.window_partition_by
            )
        elif plan.window_kind == String("lag"):
            return lag(
                df,
                plan.window_value_col,
                plan.window_param_int,
                plan.window_order_by,
                plan.window_partition_by,
                plan.window_new_name
            )
        elif plan.window_kind == String("lead"):
            return lead(
                df,
                plan.window_value_col,
                plan.window_param_int,
                plan.window_order_by,
                plan.window_partition_by,
                plan.window_new_name
            )
        elif plan.window_kind == String("cum_sum"):
            return cum_sum(
                df,
                plan.window_value_col,
                plan.window_order_by,
                plan.window_partition_by,
                plan.window_new_name
            )
        elif plan.window_kind == String("cum_mean"):
            return cum_mean(
                df,
                plan.window_value_col,
                plan.window_order_by,
                plan.window_partition_by,
                plan.window_new_name
            )
        else:
# dense_rank as a safe fallback
            return dense_rank(
                df,
                plan.window_order_by,
                plan.window_partition_by,
                plan.window_new_name
            )

# Aggregation
# Executes global or grouped aggregations. If no keys are provided, a single-row
# DataFrame is returned with one column per aggregation.
fn exec_aggregate(df: DataFrame, keys: List[String], aggs: List[AggSpec]) -> DataFrame:
    if len(keys) == 0:
# Global aggregations -> single-row DataFrame
        var out_names = List[String]()
        var out_cols = List[Column]()
        var i = 0
        var m = len(aggs)
        while i < m:
            var a = aggs[i]
            out_names.push_back(a.alias)
            var val = agg_on_col(df, a)
# Wrap the scalar in a one-element f64 series
            out_cols.push_back(
                Column.from_f64(SeriesF64(a.alias, List[Float64]([val])))
            )
            i += 1
        return DataFrame(out_names, out_cols)

# Grouped aggregations
    return groupby_multi(df, keys, aggs)

# Group-by over multiple key columns.
# Strategy:
# 1) Build composite string keys per row (stable delimiter).
# 2) Find unique composite keys and assign group ids (gids).
# 3) For each group, collect row indices and reduce requested aggregates.
# 4) Expand key columns back as string columns + append aggregate columns.
fn groupby_multi(df: DataFrame, keys: List[String], aggs: List[AggSpec]) -> DataFrame:
    var n = df.nrows()

# Map key col_names -> column indices
    var key_cols = List[Int]()
    var ki = 0
    while ki < len(keys):
        key_cols.push_back(df.get_column_index(keys[ki]))
        ki += 1

    var comp_keys = List[String]()
    var i = 0
    while i < n:
        var s = String("")
        var k = 0
        while k < len(key_cols):
            var cidx = key_cols[k]
            if cidx >= 0:
                s = s + df.cols[cidx].value()_str(i)
# Stable unit separator between components
            s = s + String("␟")
            k += 1
        comp_keys.push_back(s)
        i += 1

# Unique keys and group ids (gids)
    var uniq = List[String]()
    var gids = List[Int]()
    i = 0
    while i < n:
        var s = comp_keys[i]
        var gid = -1
        var j = 0
        while j < len(uniq):
            if uniq[j] == s:
                gid = j
                break
            j += 1
        if gid < 0:
            uniq.push_back(s)
            gid = len(uniq) - 1
        gids.push_back(gid)
        i += 1

    var G = len(uniq)

# Output col_names/columns
    var out_names = List[String]()
    var out_cols = List[Column]()

# Expand group keys back as columns (string-typed)
    ki = 0
    while ki < len(keys):
        out_names.push_back(keys[ki])
        var vals = List[String]()
        var g = 0
        while g < G:
# find a representative row r within group g for key extraction
            var r = 0
            var found = String("")
            while r < n:
                if gids[r] == g:
                    var cidx = key_cols[ki]
                    found = df.cols[cidx].value()_str(r)
                    break
                r += 1
            vals.push_back(found)
            g += 1
        out_cols.push_back(Column.from_str(SeriesStr(keys[ki], vals)))
        ki += 1

# Compute aggregates per group
    var ai = 0
    while ai < len(aggs):
        var a = aggs[ai]
        var agg_vals = List[Float64]()
        var g = 0
        while g < G:
# Collect row indices in this group
            var idxs = List[Int]()
            var r = 0
            while r < n:
                if gids[r] == g:
                    idxs.push_back(r)
                r += 1
# Reduce over those rows
            agg_vals.push_back(agg_on_col_rows(df, a, idxs))
            g += 1
        out_names.push_back(a.alias)
        out_cols.push_back(Column.from_f64(SeriesF64(a.alias, agg_vals)))
        ai += 1

    return DataFrame(out_names, out_cols)

# Reduce an aggregation over a whole column (global aggregation).
# Supported ops: "sum", "mean", "count". Unknown ops fall back to "sum".
fn agg_on_col(df: DataFrame, a: AggSpec) -> Float64:
    var col = df.get_column(a.column)
    var n = col.len()

    var s: Float64 = 0.0
    var c = 0
    var i = 0
    while i < n:
        if col.is_valid(i):
            s = s + col.as_f64_or_nan(i)
            c += 1
        i += 1

    if a.op == String("sum"):
        return s
    elif a.op == String("mean"):
        if c == 0:
            return 0.0
        return s / Float64(c)
    elif a.op == String("count"):
        return Float64(c)

# Fallback to sum for unknown ops to avoid exceptions
    return s

# Reduce an aggregation over selected row indices of a column (grouped case).
# Supported ops: "sum", "mean", "count". Unknown ops fall back to "sum".
fn agg_on_col_rows(df: DataFrame, a: AggSpec, idxs: List[Int]) -> Float64:
    var col = df.get_column(a.column)

    var s: Float64 = 0.0
    var c = 0
    var i = 0
    var L = len(idxs)
    while i < L:
        var j = idxs[i]
        if j >= 0 and j < col.len() and col.is_valid(j):
            s = s + col.as_f64_or_nan(j)
            c += 1
        i += 1

    if a.op == String("sum"):
        return s
    elif a.op == String("mean"):
        if c == 0:
            return 0.0
        return s / Float64(c)
    elif a.op == String("count"):
        return Float64(c)

# Fallback to sum for unknown ops
    return s
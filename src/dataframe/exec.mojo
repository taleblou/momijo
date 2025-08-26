# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Physical execution engine for the minimal plan including Aggregate

from momijo.dataframe.logical_plan import LogicalPlan, PlanKind, AggSpec
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.expr import Expr, filter as df_filter

fn execute(plan: LogicalPlan) -> DataFrame:
    if plan.kind == PlanKind.SCAN:
        return plan.df!
    if plan.kind == PlanKind.FILTER:
        let input_df = execute(plan.input!)
        return df_filter(input_df, plan.filter_expr!)
    if plan.kind == PlanKind.PROJECT:
        let input_df = execute(plan.input!)
        return project_columns(input_df, plan.project_cols)
    if plan.kind == PlanKind.AGGREGATE:
        let input_df = execute(plan.input!)
        return exec_aggregate(input_df, plan.group_key!, plan.aggs)
    # Fallback
    return DataFrame([])

fn project_columns(df: DataFrame, cols: List[String]) -> DataFrame:
    return df.select_columns(cols)

struct AggAcc(Copyable, Movable):
    var count: Int
    var sum: Float64

    fn __init__(out self):
        self.count = 0
        self.sum = 0.0

fn exec_aggregate(df: DataFrame, by: String, aggs: List[AggSpec]) -> DataFrame:
    # Single key (Float64) grouping: build hash map key -> row indices (or accumulators)
    let key_col = df.get_column(by)
    # Prepare accumulators per output column per key
    # For simplicity, we maintain Dict[key -> Dict[out_col -> AggAcc]]
    var table = Dict[Float64, Dict[String, AggAcc]]()

    # Iterate rows
    let n = df.height()
    for i in range(0, n):
        if not key_col.validity.is_set(i): 
            continue
        let key = key_col.values[i]
        if not table.contains(key):
            table[key] = Dict[String, AggAcc]()
            # initialize accumulators per agg output
            for a in aggs:
                table[key][a.output_col] = AggAcc()

        for a in aggs:
            if a.agg_func == "count":
                table[key][a.output_col].count += 1
            else:
                let col = df.get_column(a.input_col)
                if col.validity.is_set(i):
                    table[key][a.output_col].count += 1
                    table[key][a.output_col].sum += col.values[i]

    # Build result columns: group key + each agg output
    var keys = List[Float64]()
    for k in table.keys():
        keys.append(k)

    # Sort keys for deterministic output (optional)
    keys.sort()

    var out_cols = List[SeriesF64]()
    var key_vals = List[Float64]()
    for k in keys: key_vals.append(k)
    out_cols.append(SeriesF64(by, key_vals))

    for a in aggs:
        var vals = List[Float64]()
        for k in keys:
            let acc = table[k][a.output_col]
            if a.agg_func == "sum":
                vals.append(acc.sum)
            elif a.agg_func == "mean":
                vals.append(acc.sum / Float64(acc.count) if acc.count > 0 else 0.0)
            elif a.agg_func == "count":
                vals.append(Float64(acc.count))
            else:
                assert(False, "Unknown agg func: " + a.agg_func)
        out_cols.append(SeriesF64(a.output_col, vals))

    return DataFrame(out_cols)

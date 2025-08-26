# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Logical plan nodes for a minimal DataFrame engine with Scan/Filter/Project/Aggregate

from momijo.dataframe.frame import DataFrame
from momijo.dataframe.expr import Expr

enum PlanKind:
    SCAN
    FILTER
    PROJECT
    AGGREGATE

struct AggSpec(Copyable, Movable):
    var input_col: String
    var agg_func: String   # "sum","mean","count"
    var output_col: String

    fn __init__(out self, input_col: String, agg_func: String, output_col: String):
        self.input_col = input_col
        self.agg_func = agg_func
        self.output_col = output_col

struct LogicalPlan(Copyable, Movable):
    var kind: PlanKind
    var input: Optional[Pointer[LogicalPlan]]
    var df: Optional[DataFrame]     # for SCAN
    var filter_expr: Optional[Expr] # for FILTER
    var project_cols: List[String]  # for PROJECT
    var group_key: Optional[String] # for AGGREGATE (single key)
    var aggs: List[AggSpec]         # for AGGREGATE

    fn __init__(out self, kind: PlanKind):
        self.kind = kind
        self.input = None
        self.df = None
        self.filter_expr = None
        self.project_cols = []
        self.group_key = None
        self.aggs = []

fn scan(df: DataFrame) -> LogicalPlan:
    var p = LogicalPlan(PlanKind.SCAN)
    p.df = df
    return p

fn filter(input: LogicalPlan, e: Expr) -> LogicalPlan:
    var p = LogicalPlan(PlanKind.FILTER)
    p.input = &input
    p.filter_expr = e
    return p

fn project(input: LogicalPlan, cols: List[String]) -> LogicalPlan:
    var p = LogicalPlan(PlanKind.PROJECT)
    p.input = &input
    p.project_cols = cols
    return p

fn aggregate(input: LogicalPlan, by: String, aggs: List[AggSpec]) -> LogicalPlan:
    var p = LogicalPlan(PlanKind.AGGREGATE)
    p.input = &input
    p.group_key = by
    p.aggs = aggs
    return p

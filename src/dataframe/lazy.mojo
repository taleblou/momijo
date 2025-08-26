# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# LazyFrame wrapping a LogicalPlan; now supports select/filter/aggregate

from momijo.dataframe.logical_plan import LogicalPlan, scan as lp_scan, filter as lp_filter, project as lp_project, aggregate as lp_agg, AggSpec
from momijo.dataframe.optimizer import optimize
from momijo.dataframe.exec import execute
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.expr import Expr, Pred

struct LazyFrame(Copyable, Movable):
    var plan: LogicalPlan

    fn __init__(out self, plan: LogicalPlan):
        self.plan = plan

    fn filter(self, e: Expr) -> LazyFrame:
        return LazyFrame(lp_filter(self.plan, e))

    fn select(self, cols: List[String]) -> LazyFrame:
        return LazyFrame(lp_project(self.plan, cols))

    fn groupby(self, by: String) -> LazyGroupBy:
        return LazyGroupBy(self.plan, by)

    fn collect(self) -> DataFrame:
        let opt = optimize(self.plan)
        return execute(opt)

struct LazyGroupBy(Copyable, Movable):
    var input_plan: LogicalPlan
    var by: String

    fn __init__(out self, input_plan: LogicalPlan, by: String):
        self.input_plan = input_plan
        self.by = by

    fn agg(self, aggs: List[AggSpec]) -> LazyFrame:
        return LazyFrame(lp_agg(self.input_plan, self.by, aggs))

fn from_df(df: DataFrame) -> LazyFrame:
    return LazyFrame(lp_scan(df))

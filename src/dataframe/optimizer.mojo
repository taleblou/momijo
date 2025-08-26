# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Rule-based optimizer: push projection below filter; prune projection against aggregate needs

from momijo.dataframe.logical_plan import LogicalPlan, PlanKind, filter as lp_filter, project as lp_project, aggregate as lp_agg, AggSpec
from momijo.dataframe.expr import columns_referenced

fn optimize(plan: LogicalPlan) -> LogicalPlan:
    # PROJECT over FILTER: pushdown
    if plan.kind == PlanKind.PROJECT and plan.input is not None:
        let child = plan.input!
        if child.kind == PlanKind.FILTER and child.filter_expr is not None:
            var needed = Set[String]()
            # columns needed for final projection
            for c in plan.project_cols:
                needed.insert(c)
            # plus columns used in filter
            for c in columns_referenced(child.filter_expr!):
                needed.insert(c)
            # reconstruct: PROJECT(needed) -> FILTER -> input
            var below = lp_project(child.input!, List[String](needed)) if child.input is not None else child
            var newf = lp_filter(below, child.filter_expr!)
            return lp_project(newf, plan.project_cols)

    # PROJECT over AGGREGATE: prune to only requested outputs
    if plan.kind == PlanKind.PROJECT and plan.input is not None:
        let child = plan.input!
        if child.kind == PlanKind.AGGREGATE:
            # project only requested output cols that exist after agg
            var cols = List[String]()
            if child.group_key is not None and plan.project_cols.contains(child.group_key!):
                cols.append(child.group_key!)
            for a in child.aggs:
                if plan.project_cols.contains(a.output_col):
                    cols.append(a.output_col)
            return lp_project(child, cols)

    return plan

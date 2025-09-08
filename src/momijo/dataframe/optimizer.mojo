# Project:      Momijo
# Module:       src.momijo.dataframe.optimizer
# File:         optimizer.mojo
# Path:         src/momijo/dataframe/optimizer.mojo
#
# Description:  src.momijo.dataframe.optimizer — focused Momijo functionality with a stable public API.
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
#   - Key functions: optimize_once, optimize


from momijo.dataframe.logical_plan import (

    LogicalPlan,
    PLAN_PROJECT,
    PLAN_FILTER
)

# --- one pass of top-level rewrite ---
fn optimize_once(p: LogicalPlan) -> LogicalPlan:
    # Rule 1: Project over Filter  -> swap
    if p.kind == PLAN_PROJECT and p.child.kind == PLAN_FILTER:
        # p = Project(Filter(X, e), cols)  ==>  Filter(Project(X, cols), e)
        var filter_child = p.child.child
        var filter_expr = p.child.expr
        var new_child = LogicalPlan.project(filter_child, p.columns)
        return LogicalPlan.filter(new_child, filter_expr)

    # Rule 2: Filter over Project -> swap
    if p.kind == PLAN_FILTER and p.child.kind == PLAN_PROJECT:
        # p = Filter(Project(X, cols), e)  ==>  Project(Filter(X, e), cols)
        var proj_child = p.child.child
        var proj_cols = p.child.columns
        var new_child = LogicalPlan.filter(proj_child, p.expr)
        return LogicalPlan.project(new_child, proj_cols)

    # Rule 3: Collapse double Project
    if p.kind == PLAN_PROJECT and p.child.kind == PLAN_PROJECT:
        # Project(Project(X, cols1), cols2) -> Project(X, cols2)
        var inner_child = p.child.child
        return LogicalPlan.project(inner_child, p.columns)

    # No change at top-level in this pass
    return p

# --- run a few passes to propagate rewrites ---
fn optimize(plan: LogicalPlan) -> LogicalPlan:
    var cur = plan
    var i = 0
    # Run a small fixed number of passes. This helps push rules deeper
    # without requiring full recursion/rebuild for all node kinds.
    while i < 6:
        var nxt = optimize_once(cur)
        # If no change at top, further passes may still catch a new pattern
        # exposed by the previous rewrite one level down, so we just continue.
        cur = nxt
        i += 1
    return cur
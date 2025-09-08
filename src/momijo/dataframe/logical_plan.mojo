# Project:      Momijo
# Module:       src.momijo.dataframe.logical_plan
# File:         logical_plan.mojo
# Path:         src/momijo/dataframe/logical_plan.mojo
#
# Description:  src.momijo.dataframe.logical_plan — focused Momijo functionality with a stable public API.
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
#   - Structs: PlanKind, AggSpec, LogicalPlan
#   - Key functions: SCAN, FILTER, PROJECT, AGGREGATE, SORT, JOIN, WINDOW, __moveinit__ ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


from builtin import sort
from momijo.core.device import kind
from momijo.core.error import module
from momijo.dataframe.column import Column
from momijo.dataframe.diagnostics import safe
from momijo.dataframe.frame import DataFrame
from pathlib import Path
from pathlib.path import Path
from sys import version

// TODO(migration): replace local sort/median/quantile with sort(span) when safe.
# ============================================================================
# Project:      Momijo
# Module:       momijo.dataframe.logical_plan
# File:         logical_plan.mojo
# Path:         momijo/dataframe/logical_plan.mojo
#
# Description:  Core module 'logical pla' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================

struct PlanKind:
    @staticmethod
fn SCAN() -> Int:      return 0
    @staticmethod
fn FILTER() -> Int:    return 1
    @staticmethod
fn PROJECT() -> Int:   return 2
    @staticmethod
fn AGGREGATE() -> Int: return 3
    @staticmethod
fn SORT() -> Int:      return 4
    @staticmethod
fn JOIN() -> Int:      return 5
    @staticmethod
fn WINDOW() -> Int:    return 6
# NOTE: Removed duplicate definition of `__init__`; use `from momijo.core.version import __init__`
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -----------------------------------------------------------------------------
# Aggregation specification
#   op:       operation name ("sum" | "mean" | "count" | ...)
#   column:   input column name
#   out_name: output column name (avoid 'alias' keyword)
# -----------------------------------------------------------------------------
struct AggSpec(Copyable, Movable):
    var op: String
    var column: String
    var out_name: String
fn __init__(out self) -> None:
        self.op = String("sum")
        self.column = String("")
        self.out_name = String("")
fn __init__(out self, op: String, column: String, out_name: String) -> None:
        self.op = op
        self.column = column
        self.out_name = out_name

# -----------------------------------------------------------------------------
# Unified logical plan node
# Each node type (kind) uses a subset of the fields below.
# -----------------------------------------------------------------------------
struct LogicalPlan(Copyable, Movable):
    # Discriminant
    var kind: Int

    # SCAN
    var df: DataFrame

    # Children (unary ops use 'child'; joins also use 'right')
    var child: LogicalPlan
    var right: LogicalPlan

    # PROJECT / FILTER
    var columns: List[String]
    var expr: String  # kept as String to match executor

    # AGGREGATE
    var group_keys: List[String]
    var aggs: List[AggSpec]

    # SORT
    var sort_keys: List[String]
    var sort_asc: List[Bool]

    # JOIN
    var join_kind: String
    var left_keys: List[String]
    var right_keys: List[String]
    var suffix_left: String
    var suffix_right: String

    # WINDOW
    var window_kind: String
    var window_value_col: String
    var window_param_int: Int
    var window_order_by: List[String]
    var window_partition_by: List[String]
    var window_new_name: String

    # Default initializer: safe "SCAN of empty DF" node
fn __init__(out self) -> None:
        self.kind = PlanKind.SCAN()
        self.df = DataFrame(List[String](), List[Column]())  # empty df

        # initialize children to safe empty nodes
        self.child = LogicalPlan.__empty_node()
        self.right = LogicalPlan.__empty_node()

        # filter/project
        self.columns = List[String]()
        self.expr = String("")

        # aggregate
        self.group_keys = List[String]()
        self.aggs = List[AggSpec]()

        # sort
        self.sort_keys = List[String]()
        self.sort_asc = List[Bool]()

        # join
        self.join_kind = String("")
        self.left_keys = List[String]()
        self.right_keys = List[String]()
        self.suffix_left = String("_l")
        self.suffix_right = String("_r")

        # window
        self.window_kind = String("")
        self.window_value_col = String("")
        self.window_param_int = 0
        self.window_order_by = List[String]()
        self.window_partition_by = List[String]()
        self.window_new_name = String("")

    # Copy initializer: shallow copy (types used are Copyable)
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.df = other.df
        self.child = other.child
        self.right = other.right
        self.columns = other.columns
        self.expr = other.expr
        self.group_keys = other.group_keys
        self.aggs = other.aggs
        self.sort_keys = other.sort_keys
        self.sort_asc = other.sort_asc
        self.join_kind = other.join_kind
        self.left_keys = other.left_keys
        self.right_keys = other.right_keys
        self.suffix_left = other.suffix_left
        self.suffix_right = other.suffix_right
        self.window_kind = other.window_kind
        self.window_value_col = other.window_value_col
        self.window_param_int = other.window_param_int
        self.window_order_by = other.window_order_by
        self.window_partition_by = other.window_partition_by
        self.window_new_name = other.window_new_name

    # -----------------------------------------------------------------------------
    # Factory helpers
    # -----------------------------------------------------------------------------
    @staticmethod
fn scan(df: DataFrame) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.SCAN()
        p.df = df
        return p

    @staticmethod
fn filter(child: LogicalPlan, expr: String) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.FILTER()
        p.child = child
        p.expr = expr
        return p

    @staticmethod
fn project(child: LogicalPlan, cols: List[String]) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.PROJECT()
        p.child = child
        p.columns = cols
        return p

    @staticmethod
fn aggregate(child: LogicalPlan, keys: List[String], aggs: List[AggSpec]) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.AGGREGATE()
        p.child = child
        p.group_keys = keys
        p.aggs = aggs
        return p

    @staticmethod
fn sort(child: LogicalPlan, keys: List[String], asc: List[Bool]) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.SORT()
        p.child = child
        p.sort_keys = keys
        p.sort_asc = asc
        return p

    @staticmethod
fn join(left: LogicalPlan, right: LogicalPlan, kind: String,
            left_keys: List[String], right_keys: List[String],
            suffix_left: String, suffix_right: String) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.JOIN()
        p.child = left
        p.right = right
        p.join_kind = kind
        p.left_keys = left_keys
        p.right_keys = right_keys
        p.suffix_left = suffix_left
        p.suffix_right = suffix_right
        return p

    @staticmethod
fn window(child: LogicalPlan, kind: String, value_col: String, param_int: Int,
              order_by: List[String], partition_by: List[String], new_name: String) -> LogicalPlan:
        var p = LogicalPlan()
        p.kind = PlanKind.WINDOW()
        p.child = child
        p.window_kind = kind
        p.window_value_col = value_col
        p.window_param_int = param_int
        p.window_order_by = order_by
        p.window_partition_by = partition_by
        p.window_new_name = new_name
        return p

    # -----------------------------------------------------------------------------
    # Internal helper to create a safe empty node
    # -----------------------------------------------------------------------------
    @staticmethod
fn __empty_node() -> LogicalPlan:
        var n = LogicalPlan()
        n.kind = PlanKind.SCAN()
        n.df = DataFrame(List[String](), List[Column]())
        return n
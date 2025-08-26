# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Expression system: predicates combined via AND/OR (flat; no nesting)
# Supports ops: >, >=, <, <=, ==, != on numeric columns (Float64)

from momijo.dataframe.frame import DataFrame
from momijo.dataframe.series_f64 import SeriesF64
from momijo.arrow_core.bitmap import Bitmap

struct Pred(Copyable, Movable):
    var column: String
    var op: String   # one of >,>=,<,<=,==,!=
    var value: Float64

    fn __init__(out self, column: String, op: String, value: Float64):
        self.column = column
        self.op = op
        self.value = value

struct Expr(Copyable, Movable):
    # mode in {"single", "and", "or"}
    var mode: String
    var preds: List[Pred]

    fn single(out self, p: Pred):
        self.mode = "single"
        self.preds = [p]

    fn all(out self, preds: List[Pred]):
        self.mode = "and"
        self.preds = preds

    fn any(out self, preds: List[Pred]):
        self.mode = "or"
        self.preds = preds

fn columns_referenced(e: Expr) -> List[String]:
    var s = Set[String]()
    for p in e.preds:
        s.insert(p.column)
    return List[String](s)

fn _eval_pred(val: Float64, p: Pred) -> Bool:
    if p.op == ">": return val > p.value
    if p.op == ">=": return val >= p.value
    if p.op == "<": return val < p.value
    if p.op == "<=": return val <= p.value
    if p.op == "==": return val == p.value
    if p.op == "!=": return val != p.value
    assert(False, "Unsupported op: " + p.op)
    return False

fn filter(df: DataFrame, e: Expr) -> DataFrame:
    let n = df.height()
    var mask = Bitmap(nbits=n, all_valid=False)

    if e.mode == "single":
        let p = e.preds[0]
        let col = df.get_column(p.column)
        for i in range(0, n):
            if col.validity.is_set(i) and _eval_pred(col.values[i], p):
                mask.set(i)

    elif e.mode == "and":
        # row satisfies all preds
        # We could prefetch columns for efficiency
        var cols = Dict[String, SeriesF64]()
        for p in e.preds:
            cols[p.column] = df.get_column(p.column)
        for i in range(0, n):
            var keep = True
            for p in e.preds:
                let c = cols[p.column]
                if not (c.validity.is_set(i) and _eval_pred(c.values[i], p)):
                    keep = False
                    break
            if keep: mask.set(i)

    elif e.mode == "or":
        var cols = Dict[String, SeriesF64]()
        for p in e.preds:
            cols[p.column] = df.get_column(p.column)
        for i in range(0, n):
            var keep = False
            for p in e.preds:
                let c = cols[p.column]
                if c.validity.is_set(i) and _eval_pred(c.values[i], p):
                    keep = True
                    break
            if keep: mask.set(i)
    else:
        assert(False, "Unknown Expr.mode: " + e.mode)

    return df.filter_by_mask(mask)

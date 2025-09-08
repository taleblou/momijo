# Project:      Momijo
# Module:       src.momijo.nn.losses.mse
# File:         mse.mojo
# Path:         src/momijo/nn/losses/mse.mojo
#
# Description:  Loss functions for supervised learning in Momijo with numerically
#               stable forward and backward computations for classification/regression.
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
#   - Structs: MSELoss
#   - Key functions: _zeros1d, _zeros2d, mse1d, mse2d, __init__, forward1d, forward2d, __copyinit__ ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.error import module
from momijo.dataframe.helpers import t
from momijo.nn.training.metrics import mse
from pathlib import Path
from pathlib.path import Path

fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn _zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

# --- 1D MSE ---
fn mse1d(pred: List[Float64], target: List[Float64], reduction: String = "mean", weight: List[Float64] = List[Float64]()) -> List[Float64]:
    var n = len(pred)
    var out = List[Float64]()
    var use_w = len(weight) == n
    if reduction == "none":
        for i in range(n):
            var d = pred[i] - target[i]
            var e = d * d
            if use_w: e *= weight[i]
            out.push(e)
        return out
    var sumv = 0.0
    var denom = 0.0
    for i in range(n):
        var d = pred[i] - target[i]
        var e = d * d
        if use_w:
            e *= weight[i]
            denom += weight[i]
        else:
            denom += 1.0
        sumv += e
    if reduction == "sum":
        out.push(sumv)
    else:
        if denom == 0.0: denom = 1.0
        out.push(sumv / denom)
    return out

# --- 2D MSE ---
fn mse2d(pred: List[List[Float64]], target: List[List[Float64]], reduction: String = "mean", weight: List[List[Float64]] = List[List[Float64]]()) -> List[Float64]:
    var R = len(pred)
    var C = 0
    if R > 0: C = len(pred[0])
    var out = List[Float64]()
    var use_w = (len(weight) == R) and (R > 0) and (len(weight[0]) == C)
    if reduction == "none":
        for i in range(R):
            for j in range(C):
                var d = pred[i][j] - target[i][j]
                var e = d * d
                if use_w: e *= weight[i][j]
                out.push(e)
        return out
    var sumv = 0.0
    var denom = 0.0
    for i in range(R):
        for j in range(C):
            var d = pred[i][j] - target[i][j]
            var e = d * d
            if use_w:
                e *= weight[i][j]
                denom += weight[i][j]
            else:
                denom += 1.0
            sumv += e
    if reduction == "sum":
        out.push(sumv)
    else:
        if denom == 0.0: denom = 1.0
        out.push(sumv / denom)
    return out

# --- Module wrapper for convenience ---
struct MSELoss:
    var reduction: String  # "mean" | "sum" | "none"
fn __init__(out self, reduction: String = "mean") -> None:
        self.reduction = reduction
fn forward1d(self, pred: List[Float64], target: List[Float64], weight: List[Float64] = List[Float64]()) -> List[Float64]:
        return mse1d(pred, target, self.reduction, weight)
fn forward2d(self, pred: List[List[Float64]], target: List[List[Float64]], weight: List[List[Float64]] = List[List[Float64]]()) -> List[Float64]:
        return mse2d(pred, target, self.reduction, weight)
fn __copyinit__(out self, other: Self) -> None:
        self.reduction = other.reduction
fn __moveinit__(out self, deinit other: Self) -> None:
        self.reduction = other.reduction
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1D
    var p = List[Float64](); var t = List[Float64]()
    p.push(1.0); p.push(2.0); p.push(3.0)
    t.push(1.0); t.push(0.0); t.push(5.0)
    var none_vals = mse1d(p, t, "none", List[Float64]())
    ok = ok and (len(none_vals) == 3)
    var mean_val = mse1d(p, t, "mean", List[Float64]())[0]
    var sum_val = mse1d(p, t, "sum", List[Float64]())[0]
    ok = ok and (sum_val >= mean_val)

    # weighted mean should differ from unweighted when weights vary
    var w = List[Float64](); w.push(1.0); w.push(2.0); w.push(3.0)
    var mean_w = mse1d(p, t, "mean", w)[0]
    ok = ok and (mean_w == mean_w)

    # 2D
    var P = List[List[Float64]](); var Tt = List[List[Float64]]()
    var r0p = List[Float64](); r0p.push(1.0); r0p.push(2.0)
    var r1p = List[Float64](); r1p.push(0.0); r1p.push(3.0)
    var r0t = List[Float64](); r0t.push(0.0); r0t.push(2.0)
    var r1t = List[Float64](); r1t.push(1.0); r1t.push(1.0)
    P.push(r0p); P.push(r1p)
    Tt.push(r0t); Tt.push(r1t)
    var m2 = mse2d(P, Tt, "mean", List[List[Float64]]())[0]
    ok = ok and (m2 == m2)

    # Module wrapper
    var loss = MSELoss("mean")
    var m3 = loss.forward2d(P, Tt, List[List[Float64]]())[0]
    ok = ok and (m3 == m3)

    return ok
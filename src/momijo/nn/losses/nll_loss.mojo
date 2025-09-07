# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.nn.losses
# File: src/momijo/nn/losses/nll_loss.mojo

from momijo.core.error import module
from momijo.dataframe.expr import single
from momijo.utils.random import sample
from momijo.utils.result import g
from pathlib import Path
from pathlib.path import Path

fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

# single example loss; returns (loss, denom_weight)
fn _nll_row(log_probs: List[Float64], target_idx: Int, class_weight: List[Float64], ignore_index: Int) -> (Float64, Float64):
    var C = len(log_probs)
    if C == 0:
        return (0.0, 0.0)
    if target_idx < 0 or target_idx >= C:
        # out of range target -> treat as ignored
        return (0.0, 0.0)
    if target_idx == ignore_index:
        return (0.0, 0.0)
    var w = 1.0
    if len(class_weight) == C:
        w = class_weight[target_idx]
    var loss = -log_probs[target_idx] * w
    return (loss, w)

# --- 1D API ---
fn nll_loss1d(log_probs: List[Float64], target: Int, reduction: String = "mean", class_weight: List[Float64] = List[Float64](), ignore_index: Int = -100) -> List[Float64]:
    var (l, w) = _nll_row(log_probs, target, class_weight, ignore_index)
    var out = List[Float64]()
    if reduction == "none":
        out.push(l)
        return out
    if reduction == "sum":
        out.push(l)
        return out
    # mean
    var denom = w
    if denom == 0.0: denom = 1.0
    out.push(l / denom)
    return out

# --- 2D API: log_probs [N,C], targets indices [N] ---
fn nll_loss2d(log_probs: List[List[Float64]], targets: List[Int], reduction: String = "mean", class_weight: List[Float64] = List[Float64](), ignore_index: Int = -100) -> List[Float64]:
    var N = len(log_probs)
    var out = List[Float64]()
    if reduction == "none":
        for n in range(N):
            var (ln, _) = _nll_row(log_probs[n], targets[n], class_weight, ignore_index)
            out.push(ln)
        return out
    var sumv = 0.0
    var denom = 0.0
    for n in range(N):
        var (ln, wn) = _nll_row(log_probs[n], targets[n], class_weight, ignore_index)
        sumv += ln
        denom += wn
    if reduction == "sum":
        out.push(sumv)
    else:
        if denom == 0.0: denom = 1.0
        out.push(sumv / denom)
    return out

# --- Module wrapper ---
struct NLLLoss:
    var reduction: String
    var ignore_index: Int
    var class_weight: List[Float64]  # optional per-class weights
fn __init__(out self, reduction: String = "mean", ignore_index: Int = -100, class_weight: List[Float64] = List[Float64]()):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.class_weight = class_weight
fn forward1d(self, log_probs: List[Float64], target: Int) -> List[Float64]:
        return nll_loss1d(log_probs, target, self.reduction, self.class_weight, self.ignore_index)
fn forward2d(self, log_probs: List[List[Float64]], targets: List[Int]) -> List[Float64]:
        return nll_loss2d(log_probs, targets, self.reduction, self.class_weight, self.ignore_index)
fn __copyinit__(out self, other: Self) -> None:
        self.reduction = other.reduction
        self.ignore_index = other.ignore_index
        self.class_weight = other.class_weight
fn __moveinit__(out self, deinit other: Self) -> None:
        self.reduction = other.reduction
        self.ignore_index = other.ignore_index
        self.class_weight = other.class_weight
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Single sample, 3-class
    var lp = List[Float64](); lp.push(-2.0); lp.push(-0.1); lp.push(-3.0)  # e.g., log_softmax outputs
    var l1 = nll_loss1d(lp, 1, "mean", List[Float64](), -100)[0]
    ok = ok and (l1 >= 0.0)

    # Batch
    var LP = List[List[Float64]]()
    LP.push(List[Float64]([-1.0,-2.0,-3.0]))
    LP.push(List[Float64]([-0.5,-1.5,-2.5]))
    var T = List[Int]([0, 2])
    var l2 = nll_loss2d(LP, T, "sum", List[Float64](), -100)[0]
    ok = ok and (l2 >= 0.0)

    # Weights + ignore_index
    var W = List[Float64]([1.0, 2.0, 3.0])
    var T2 = List[Int]([0, -100])  # second ignored
    var l3 = nll_loss2d(LP, T2, "mean", W, -100)[0]
    ok = ok and (l3 == l3)

    # none reduction returns vector
    var v = nll_loss2d(LP, T, "none", List[Float64](), -100)
    ok = ok and (len(v) == 2)

    # Module wrapper
    var loss = NLLLoss("mean", -100, W)
    var l4 = loss.forward2d(LP, T)[0]
    ok = ok and (l4 == l4)

    return ok
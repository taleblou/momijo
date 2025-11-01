# Project:      Momijo
# Module:       learn.engine.evaluator
# File:         engine/evaluator.mojo
# Path:         src/momijo/learn/engine/evaluator.mojo
#
# Description:  Evaluation loop utilities for Momijo Learn. Provides a backend-agnostic
#               Evaluator that iterates over a DataLoader-like object, computes loss
#               and user-provided metrics, and returns a printable EvalResult.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand

from collections.list import List
from pathlib.path import Path
from momijo.tensor.tensor import Tensor   # ← central tensor import (per project rule)

# ------------------------------
# EvalResult: holds scalar stats
# ------------------------------
struct EvalResult:
    var samples: Int
    var batches: Int
    var loss_avg: Float64
    var metric_names: List[String]
    var metric_avgs: List[Float64]

    fn __init__(out self):
        self.samples = 0
        self.batches = 0
        self.loss_avg = 0.0
        self.metric_names = List[String]()
        self.metric_avgs = List[Float64]()

    fn __str__(self) -> String:
        var s = String("EvalResult(")
        s = s + String("samples=") + String(self.samples)
        s = s + String(", batches=") + String(self.batches)
        s = s + String(", loss_avg=") + String(self.loss_avg)
        if len(self.metric_names) > 0:
            s = s + String(", metrics={")
            var i = 0
            while i < len(self.metric_names):
                s = s + self.metric_names[i] + String(":") + String(self.metric_avgs[i])
                if i + 1 < len(self.metric_names):
                    s = s + String(", ")
                i = i + 1
            s = s + String("}")
        s = s + String(")")
        return s


# ----------------------------------------
# Internal accumulator for running averages
# ----------------------------------------
struct _RunningStats:
    var count: Int
    var sum_loss: Float64
    var metric_sums: List[Float64]

    fn __init__(out self, n_metrics: Int):
        self.count = 0
        self.sum_loss = 0.0
        self.metric_sums = List[Float64]()
        var i = 0
        while i < n_metrics:
            self.metric_sums.push_back(0.0)
            i = i + 1

    fn update(mut self, loss_val: Float64, metric_vals: List[Float64]):
        self.count = self.count + 1
        self.sum_loss = self.sum_loss + loss_val
        var i = 0
        while i < len(self.metric_sum) and i < len(metric_vals):
            self.metric_sums[i] = self.metric_sums[i] + metric_vals[i]
            i = i + 1

    fn loss_avg(self) -> Float64:
        if self.count == 0:
            return 0.0
        return self.sum_loss / Float64(self.count)

    fn metrics_avg(self) -> List[Float64]:
        var avgs = List[Float64]()
        if self.count == 0:
            var i = 0
            while i < len(self.metric_sums):
                avgs.push_back(0.0)
                i = i + 1
            return avgs
        var i2 = 0
        while i2 < len(self.metric_sums):
            avgs.push_back(self.metric_sums[i2] / Float64(self.count))
            i2 = i2 + 1
        return avgs


# ------------------------------
# Evaluator
# ------------------------------
struct Evaluator:
    var loss_fn_set: Bool
    var metric_names: List[String]

    fn __init__(out self):
        self.loss_fn_set = False
        self.metric_names = List[String]()

    # Configure a loss function (optional if model provides eval_step)
    fn with_loss(mut self, loss_name: String) -> Evaluator:
        self.loss_fn_set = True
        return self

    # Configure metric names for pretty reporting (order matters)
    fn with_metrics(mut self, names: List[String]) -> Evaluator:
        self.metric_names = names
        return self

    # Main evaluation loop.
    #   model         : object with either eval_step(batch)->(pred, loss_scalar) OR forward(x)->pred
    #   data_loader   : object with __len__() and __getitem__(idx)->batch
    #   loss_fn       : optional (pred, target)->Float64 if model has no eval_step
    #   metric_fns    : list of metric(pred, target)->Float64
    fn evaluate(
        self,
        model,
        data_loader,
        loss_fn: Optional[any] = None,
        metric_fns: List[any] = List[any](),
        verbose: Bool = False
    ) -> EvalResult:

        var n_batches = data_loader.__len__()
        var n_metrics = len(metric_fns)
        var stats = _RunningStats(n_metrics)

        var total_samples = 0
        var b = 0
        while b < n_batches:
            var batch = data_loader.__getitem__(b)

            var use_eval_step = False
            if loss_fn is None:
                use_eval_step = True

            var loss_val = 0.0
            var metric_vals = List[Float64]()

            if use_eval_step:
                # Expect: eval_step returns (pred, loss_scalar)
                var out_pair = model.eval_step(batch)
                var pred = out_pair[0]
                loss_val = out_pair[1]

                var tgt_opt = _try_get_target(batch)         # Optional[Any]
                var i = 0
                while i < n_metrics:
                    var mv = _safe_metric(metric_fns[i], pred, tgt_opt)
                    metric_vals.push_back(mv)
                    i = i + 1
            else:
                # Use forward + external loss_fn + metrics
                var xy_opt = _unpack_xy(batch)               # Optional[(x,y)]
                if xy_opt is None:
                    # If user passed custom batch, we cannot compute; keep zeros.
                    pass
                else:
                    var pair = xy_opt.value()
                    var x = pair[0]
                    var y = pair[1]
                    var pred2 = model.forward(x)
                    # loss_fn is present here by construction
                    var lf = loss_fn.value()
                    loss_val = lf(pred2, y)
                    var j = 0
                    while j < n_metrics:
                        var mv2 = _safe_metric(metric_fns[j], pred2, Optional.any(y))
                        metric_vals.push_back(mv2)
                        j = j + 1

            stats.update(loss_val, metric_vals)

            # Best-effort batch-size (Tensor-aware)
            var bs = 0
            try:
                bs = _batch_size_of(batch)
            except _:
                bs = 0
            total_samples = total_samples + bs

            if verbose:
                print(String("[eval] batch=") + String(b) + String("/") + String(n_batches - 1) + String(", loss=") + String(loss_val))

            b = b + 1

        var result = EvalResult()
        result.samples = total_samples
        result.batches = n_batches
        result.loss_avg = stats.loss_avg()

        var names = self.metric_names
        if len(names) != n_metrics:
            names = List[String]()
            var k = 0
            while k < n_metrics:
                names.push_back(String("m") + String(k))
                k = k + 1
        result.metric_names = names
        result.metric_avgs = stats.metrics_avg()
        return result


# ------------------------------
# Helpers (backend-agnostic, Tensor-aware)
# ------------------------------

# Try to unpack a (x, y) pair from a generic batch; return Optional[(x,y)]
fn _unpack_xy(batch) raises -> Optional[Tuple[any, any]]:
    # Convention: if batch is indexable with length >= 2, treat [0] as x and [1] as y
    var x = batch[0]
    var y = batch[1]
    return Optional[Tuple[any, any]]((x, y))

# Try to fetch target from an arbitrary batch; return Optional[Any]
fn _try_get_target(batch) raises -> Optional[any]:
    var y = batch[1]
    return Optional.any(y)

# Try to call a metric(pred, target_opt) safely; if it fails, return 0.0
fn _safe_metric(metric_fn, pred, target_opt: Optional[any]) -> Float64:
    var val = 0.0
    # metrics typically need target; if None, pass a dummy Optional to avoid raise
    try:
        if target_opt is None:
            val = metric_fn(pred, Optional.any(None))
        else:
            val = metric_fn(pred, target_opt.value())
    except _:
        val = 0.0
    return val

# Estimate batch size for reporting (best-effort).
# If x is a Tensor with __len__ -> first dimension, use that; otherwise 0.
fn _batch_size_of(batch) raises -> Int:
    var x = batch[0]
    # Path A: x is a Tensor[...] with __len__()
    try:
        var n = x.__len__()
        return n
    except _:
        pass
    # Path B: x is a list-like with __len__()
    try:
        var m = len(x)
        return m
    except _:
        pass
    return 0

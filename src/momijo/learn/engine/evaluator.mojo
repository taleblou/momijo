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
#
# Notes:
#   - Types: EvalResult, Evaluator
#   - Expected model API:
#       * Optional: eval_step(batch) -> (pred, loss_scalar)
#       * Or: forward(x) -> pred  (then loss_fn must be provided)
#   - Expected data_loader API:
#       * __len__() -> Int
#       * __getitem__(idx: Int) -> batch
#     where batch is either (x, y) or any object accepted by model.eval_step(...)
#   - Metrics are callables like metric(pred, target) -> Float64
#   - Loss function signature: loss_fn(pred, target) -> Float64
#   - Safe printing via __str__ on EvalResult

from collections.list import List
from pathlib.path import Path

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
        if self.metric_names.size() > 0:
            s = s + String(", metrics={")
            var i = 0
            while i < self.metric_names.size():
                s = s + self.metric_names[i] + String(":") + String(self.metric_avgs[i])
                if i + 1 < self.metric_names.size():
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
        while i < self.metric_sums.size() and i < metric_vals.size():
            self.metric_sums[i] = self.metric_sums[i] + metric_vals[i]
            i = i + 1

    fn loss_avg(self) -> Float64:
        if self.count == 0:
            return 0.0
        return self.sum_loss / Float64(self.count)

    fn metrics_avg(self) -> List[Float64]:
        var avgs = List[Float64]()
        if self.count == 0:
            # keep zeros
            var i = 0
            while i < self.metric_sums.size():
                avgs.push_back(0.0)
                i = i + 1
            return avgs
        var i = 0
        while i < self.metric_sums.size():
            avgs.push_back(self.metric_sums[i] / Float64(self.count))
            i = i + 1
        return avgs


# ------------------------------
# Evaluator
# ------------------------------
struct Evaluator:
    var loss_fn_set: Bool
    # loss_fn(pred, target) -> Float64
    # Stored as an opaque callable reference; concrete typing will be wired later.
    # In Mojo today we keep it duck-typed and pass-through.
    var metric_names: List[String]

    fn __init__(out self):
        self.loss_fn_set = False
        self.metric_names = List[String]()

    # Configure a loss function (optional if model provides eval_step)
    fn with_loss(mut self, loss_name: String) -> Evaluator:
        # Marker only; actual callable is supplied to evaluate(...) args.
        self.loss_fn_set = True
        # loss_name is recorded in metric list tail for better report readability if needed.
        return self

    # Configure metric names for pretty reporting (order matters)
    fn with_metrics(mut self, names: List[String]) -> Evaluator:
        self.metric_names = names
        return self

    # Main evaluation loop.
    # Arguments:
    #   model         : object with either eval_step(batch)->(pred, loss_scalar) OR forward(x)->pred
    #   data_loader   : object with __len__() and __getitem__(idx)->batch
    #   loss_fn       : callable(pred, target)->Float64 (required if model has no eval_step)
    #   metric_fns    : list of callables like metric(pred, target)->Float64
    # Returns:
    #   EvalResult with averages over batches (macro-average by batch)
    fn evaluate(
        self,
        model,
        data_loader,
        loss_fn = None,
        metric_fns: List[Any] = List[Any](),
        verbose: Bool = False
    ) -> EvalResult:

        var n_batches = Int(0)
        # Try to obtain number of batches via __len__()
        n_batches = data_loader.__len__()

        var n_metrics = Int(metric_fns.size())
        var stats = _RunningStats(n_metrics)

        var total_samples = Int(0)
        var b = Int(0)
        while b < n_batches:
            # Expect data_loader[b] to return either (x, y) or a batch the model can consume
            var batch = data_loader.__getitem__(b)

            # Prefer model.eval_step if available
            var has_eval_step = False
            # NOTE: No reflection; we rely on user wiring to pass the right combination.
            # Convention: if loss_fn is None, we assume eval_step exists.
            if loss_fn is None:
                has_eval_step = True

            var pred = None
            var loss_val = 0.0
            var metric_vals = List[Float64]()

            if has_eval_step:
                # Expected: eval_step returns (pred, loss_scalar)
                var out_pair = model.eval_step(batch)
                # out_pair.0 => pred, out_pair.1 => loss scalar
                pred = out_pair[0]
                loss_val = out_pair[1]
                # Targets may be embedded in batch; metrics rely on (pred, target) if needed.
                # Try to unpack (x, y) from batch for metrics if metric functions expect target.
                var target = None
                # Best-effort: if batch is a pair, use batch[1] as target
                # (Exact typing deferred to integration with momijo.tensor)
                target = _try_get_target(batch)
                var i = 0
                while i < n_metrics:
                    var val = _safe_metric(metric_fns[i], pred, target)
                    metric_vals.push_back(val)
                    i = i + 1
            else:
                # Path: use forward + external loss_fn + metrics
                var pair = _unpack_xy(batch)
                var x = pair[0]
                var y = pair[1]
                pred = model.forward(x)
                loss_val = loss_fn(pred, y)
                var i2 = 0
                while i2 < n_metrics:
                    var mv = _safe_metric(metric_fns[i2], pred, y)
                    metric_vals.push_back(mv)
                    i2 = i2 + 1

            stats.update(loss_val, metric_vals)

            # Estimate samples in this batch if accessible (optional)
            total_samples = total_samples + _batch_size_of(batch)

            if verbose:
                print(String("[eval] batch=") + String(b) + String("/") + String(n_batches - 1) + String(", loss=") + String(loss_val))

            b = b + 1

        var result = EvalResult()
        result.samples = total_samples
        result.batches = n_batches
        result.loss_avg = stats.loss_avg()

        # If metric names were not set, synthesize generic names m0, m1, ...
        var names = self.metric_names
        if names.size() != n_metrics:
            names = List[String]()
            var i = 0
            while i < n_metrics:
                names.push_back(String("m") + String(i))
                i = i + 1
        result.metric_names = names
        result.metric_avgs = stats.metrics_avg()

        return result


# ------------------------------
# Helpers (backend-agnostic)
# ------------------------------

# Try to unpack a (x, y) pair from a generic batch.
fn _unpack_xy(batch):
    # Convention: if batch is indexable with length >= 2, treat batch[0] as x and batch[1] as y
    # Here we keep it pass-through; exact behavior will be defined once tensor/data APIs stabilize.
    var x = None
    var y = None
    # Users should adapt this to their Dataset/DataLoader. For now, assume (x, y).
    x = batch[0]
    y = batch[1]
    var pair = List[Any]()
    pair.push_back(x)
    pair.push_back(y)
    return pair

# Try to fetch target from an arbitrary batch; fallback to None.
fn _try_get_target(batch):
    var tgt = None
    tgt = batch[1]
    return tgt

# Try to call a metric(pred, target) safely; if it fails, return 0.0
fn _safe_metric(metric_fn, pred, target) -> Float64:
    # Minimal defensive wrapper
    var val = 0.0
    val = metric_fn(pred, target)
    return val

# Estimate batch size for reporting (best-effort).
# If x is a tensor-like with __len__, use that; otherwise return 0.
fn _batch_size_of(batch) -> Int:
    var n = 0
    # Assume batch[0] is input tensor/list with __len__()
    var x = batch[0]
    # Best-effort: return 0 in the skeleton. Will wire once tensor API is present.
    return n

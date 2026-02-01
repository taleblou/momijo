# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.learn
# Module:       learn.metrics.streaming
# File:         src/momijo/learn/metrics/streaming.mojo
#
# Description:
#   Streaming metrics for scalar aggregation and binary classification.
#   - StreamingMetric: running mean/var/min/max/sum/count using Welford.
#   - WeightedStreamingMetric: weighted variant (West/Welford weighted).
#   - ConfusionMatrixBinary: streaming TP/FP/TN/FN and derived metrics.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types:
#       * StreamingMetric
#       * WeightedStreamingMetric
#       * ConfusionMatrixBinary
#   - Key fns:
#       * update(...), merge(...), reset(), compute()
#   - Numerics:
#       * Welford's algorithm (unweighted)
#       * West/Welford weighted algorithm (weighted)
#   - std(): returns variance for now (replace with sqrt once math facade is wired).

# -----------------------------------------------------------------------------
# Optional imports (kept precise, no wildcard)
# -----------------------------------------------------------------------------
from momijo.tensor.tensor import Tensor

# -----------------------------------------------------------------------------
# Unweighted streaming accumulator (Welford)
# -----------------------------------------------------------------------------

struct StreamingMetric:
    var _count: Int
    var _sum: Float32
    var _mean: Float32
    var _m2: Float32
    var _has_value: Bool
    var _min: Float32
    var _max: Float32

    fn __init__(out self):
        self._count = 0
        self._sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._has_value = False
        self._min = 0.0
        self._max = 0.0

    fn update(mut self, value: Float32):
        if not self._has_value:
            self._has_value = True
            self._count = 1
            self._sum = value
            self._mean = value
            self._m2 = 0.0
            self._min = value
            self._max = value
            return

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

        var n_prev = self._count
        var n_new = n_prev + 1
        var delta = value - self._mean
        var mean_new = self._mean + (delta / Float32(n_new))
        var delta2 = value - mean_new
        var m2_new = self._m2 + (delta * delta2)

        self._count = n_new
        self._sum = self._sum + value
        self._mean = mean_new
        self._m2 = m2_new

    fn merge(mut self, other: StreamingMetric):
        if not other._has_value:
            return
        if not self._has_value:
            self._has_value = True
            self._count = other._count
            self._sum = other._sum
            self._mean = other._mean
            self._m2 = other._m2
            self._min = other._min
            self._max = other._max
            return

        var n_a = self._count
        var n_b = other._count
        var n = n_a + n_b
        if n == 0:
            return

        var delta = other._mean - self._mean
        var mean_new = self._mean + (delta * (Float32(n_b) / Float32(n)))
        var m2_new = self._m2 + other._m2 + (delta * delta) * (Float32(n_a) * Float32(n_b) / Float32(n))

        self._count = n
        self._sum = self._sum + other._sum
        self._mean = mean_new
        self._m2 = m2_new

        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max

    fn reset(mut self):
        self._count = 0
        self._sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._has_value = False
        self._min = 0.0
        self._max = 0.0

    fn count(self) -> Int:
        return self._count

    fn sum(self) -> Float32:
        return self._sum

    fn mean(self) -> Float32:
        if self._count == 0:
            return 0.0
        return self._mean

    fn variance(self, unbiased: Bool = True) -> Float32:
        var n = self._count
        if n == 0:
            return 0.0
        if unbiased:
            if n < 2:
                return 0.0
            return self._m2 / Float32(n - 1)
        return self._m2 / Float32(n)

    fn std(self, unbiased: Bool = True) -> Float32:
        var v = self.variance(unbiased)
        # TODO: replace with sqrt(v) once math facade is available.
        return v

    fn min(self) -> Float32:
        if not self._has_value:
            return 0.0
        return self._min

    fn max(self) -> Float32:
        if not self._has_value:
            return 0.0
        return self._max

    # Convention: compute() returns the primary statistic (mean)
    fn compute(self) -> Float32:
        return self.mean()

    # -------- Batch adapters (list-based; safe & immediate) --------

    fn update_batch(mut self, values: List[Float32]) -> Int:
        var i = 0
        while i < len(values):
            self.update(values[i])
            i += 1
        return len(values)

# -----------------------------------------------------------------------------
# Weighted streaming accumulator (West/Welford weighted)
# -----------------------------------------------------------------------------

struct WeightedStreamingMetric:
    var _w_sum: Float32         # total weight
    var _mean: Float32          # weighted mean
    var _m2: Float32            # sum of weighted squared deviations
    var _has_value: Bool
    var _min: Float32
    var _max: Float32

    fn __init__(out self):
        self._w_sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._has_value = False
        self._min = 0.0
        self._max = 0.0

    # Update with (value, weight). Weight must be >= 0.
    fn update(mut self, value: Float32, weight: Float32):
        if weight <= 0.0:
            return

        if not self._has_value:
            self._has_value = True
            self._w_sum = weight
            self._mean = value
            self._m2 = 0.0
            self._min = value
            self._max = value
            return

        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

        var w_prev = self._w_sum
        var w_new = w_prev + weight
        var delta = value - self._mean
        var mean_new = self._mean + (weight * delta) / w_new
        var delta2 = value - mean_new
        var m2_new = self._m2 + weight * delta * delta2

        self._w_sum = w_new
        self._mean = mean_new
        self._m2 = m2_new

    # Merge two weighted accumulators
    fn merge(mut self, other: WeightedStreamingMetric):
        if not other._has_value:
            return
        if not self._has_value:
            self._has_value = True
            self._w_sum = other._w_sum
            self._mean = other._mean
            self._m2 = other._m2
            self._min = other._min
            self._max = other._max
            return

        var w_a = self._w_sum
        var w_b = other._w_sum
        var w = w_a + w_b
        if w <= 0.0:
            return

        var delta = other._mean - self._mean
        var mean_new = self._mean + (w_b * delta) / w
        var m2_new = self._m2 + other._m2 + (w_a * w_b * delta * delta) / w

        self._w_sum = w
        self._mean = mean_new
        self._m2 = m2_new

        if other._min < self._min:
            self._min = other._min
        if other._max > self._max:
            self._max = other._max

    fn reset(mut self):
        self._w_sum = 0.0
        self._mean = 0.0
        self._m2 = 0.0
        self._has_value = False
        self._min = 0.0
        self._max = 0.0

    fn weight_sum(self) -> Float32:
        return self._w_sum

    fn mean(self) -> Float32:
        if self._w_sum <= 0.0:
            return 0.0
        return self._mean

    # Weighted population variance: m2 / W
    # here we return the population-style variance by default.
    fn variance(self) -> Float32:
        if self._w_sum <= 0.0:
            return 0.0
        return self._m2 / self._w_sum

    fn std(self) -> Float32:
        var v = self.variance()
        # TODO: replace with sqrt(v) once math facade is available.
        return v

    fn min(self) -> Float32:
        if not self._has_value:
            return 0.0
        return self._min

    fn max(self) -> Float32:
        if not self._has_value:
            return 0.0
        return self._max

    fn compute(self) -> Float32:
        return self.mean()

    # -------- Batch adapters (list-based; safe & immediate) --------

    fn update_batch(mut self, values: List[Float32], weights: List[Float32]) -> Int:
        var n = len(values)
        var m = len(weights)
        var k = n if n < m else m
        var i = 0
        while i < k:
            self.update(values[i], weights[i])
            i += 1
        return k

# -----------------------------------------------------------------------------
# Binary confusion matrix and derived metrics
# -----------------------------------------------------------------------------

struct ConfusionMatrixBinary:
    var _tp: Int
    var _fp: Int
    var _tn: Int
    var _fn: Int

    fn __init__(out self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    # Update using a probability/score (Float32) and a target label (0/1).
    # A "threshold" in [0,1] maps the score to a predicted label.
    fn update(mut self, score: Float32, target: Int, threshold: Float32 = 0.5):
        var pred_label = 0
        if score >= threshold:
            pred_label = 1
        self.update_label(pred_label, target)

    # Update directly with predicted and true labels (0/1).
    fn update_label(mut self, pred_label: Int, target: Int):
        var p = pred_label != 0
        var t = target != 0
        if p and t:
            self._tp = self._tp + 1
        elif p and not t:
            self._fp = self._fp + 1
        elif not p and not t:
            self._tn = self._tn + 1
        else:
            self._fn = self._fn + 1

    fn merge(mut self, other: ConfusionMatrixBinary):
        self._tp = self._tp + other._tp
        self._fp = self._fp + other._fp
        self._tn = self._tn + other._tn
        self._fn = self._fn + other._fn

    fn reset(mut self):
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0

    fn tp(self) -> Int: return self._tp
    fn fp(self) -> Int: return self._fp
    fn tn(self) -> Int: return self._tn
    fn fn(self) -> Int: return self._fn

    fn support_pos(self) -> Int:
        return self._tp + self._fn

    fn support_neg(self) -> Int:
        return self._tn + self._fp

    fn total(self) -> Int:
        return self._tp + self._fp + self._tn + self._fn

    # Accuracy = (TP + TN) / Total
    fn accuracy(self) -> Float32:
        var tot = self.total()
        if tot == 0:
            return 0.0
        return Float32(self._tp + self._tn) / Float32(tot)

    # Precision (Positive Predictive Value) = TP / (TP + FP)
    fn precision(self) -> Float32:
        var denom = self._tp + self._fp
        if denom == 0:
            return 0.0
        return Float32(self._tp) / Float32(denom)

    # Recall (Sensitivity/TPR) = TP / (TP + FN)
    fn recall(self) -> Float32:
        var denom = self._tp + self._fn
        if denom == 0:
            return 0.0
        return Float32(self._tp) / Float32(denom)

    # Specificity (TNR) = TN / (TN + FP)
    fn specificity(self) -> Float32:
        var denom = self._tn + self._fp
        if denom == 0:
            return 0.0
        return Float32(self._tn) / Float32(denom)

    # F1 = 2 * (P * R) / (P + R)
    fn f1(self) -> Float32:
        var p = self.precision()
        var r = self.recall()
        var denom = p + r
        if denom == 0.0:
            return 0.0
        return 2.0 * p * r / denom

    # Balanced Accuracy = (TPR + TNR) / 2
    fn balanced_accuracy(self) -> Float32:
        return (self.recall() + self.specificity()) / 2.0

    # Convention: compute() returns accuracy by default for classification.
    fn compute(self) -> Float32:
        return self.accuracy()

    # -------- Batch adapters (list-based; safe & immediate) --------

    fn update_batch_scores(mut self, scores: List[Float32], targets: List[Int], threshold: Float32 = 0.5) -> Int:
        var n = len(scores)
        var m = len(targets)
        var k = n if n < m else m
        var i = 0
        while i < k:
            self.update(scores[i], targets[i], threshold)
            i += 1
        return k

    fn update_batch_labels(mut self, preds: List[Int], targets: List[Int]) -> Int:
        var n = len(preds)
        var m = len(targets)
        var k = n if n < m else m
        var i = 0
        while i < k:
            self.update_label(preds[i], targets[i])
            i += 1
        return k

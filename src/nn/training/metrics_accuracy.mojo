# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.metrics_accuracy
# Path:   src/momijo/nn/metrics_accuracy.mojo
#
# Focused accuracy utilities for classification.
# Self-contained (List-based), no external deps.
#
# Included:
#   - accuracy_from_labels(preds[Int], targets[Int])
#   - accuracy_from_scores(scores[N][C], targets[N])          # argmax
#   - topk_accuracy(scores[N][C], targets[N], k)
#   - confusion_matrix(targets[N], preds[N], num_classes=-1)  # (rows=true, cols=pred)
#   - class_accuracy_from_cm(cm) -> per-class recall (=diag/row_sum)
#   - balanced_accuracy(targets, preds, num_classes=-1)       # mean per-class recall
#   - AccuracyMeter (running average on batches)
#   - WindowedAccuracy(window) (sliding-window accuracy over last M examples)
#   - Multilabel subset accuracy & micro accuracy at threshold
#
# A self-test validates behavior on toy data.

# --------- Helpers ---------
fn zeros1d_f(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn zeros1d_i(n: Int) -> List[Int]:
    var y = List[Int]()
    for i in range(n): y.push(0)
    return y

fn zeros2d_i(r: Int, c: Int) -> List[List[Int]]:
    var y = List[List[Int]]()
    for i in range(r):
        var row = List[Int]()
        for j in range(c): row.push(0)
        y.push(row)
    return y

fn _safediv(num: Float64, den: Float64, fallback: Float64 = 0.0) -> Float64:
    if den == 0.0: return fallback
    return num / den

fn argmax1d_f(x: List[Float64]) -> Int:
    var n = len(x)
    if n == 0: return 0
    var m = x[0]
    var idx = 0
    for i in range(1, n):
        if x[i] > m:
            m = x[i]
            idx = i
    return idx

fn topk_indices_row(x: List[Float64], k: Int) -> List[Int]:
    var n = len(x)
    var kk = (k if k <= n else n)
    var taken = zeros1d_i(n)
    var out = List[Int]()
    for t in range(kk):
        var best = -1
        var bestv = -1.7976931348623157e308
        for i in range(n):
            if taken[i] == 1: continue
            var v = x[i]
            if v > bestv:
                bestv = v
                best = i
        if best < 0: break
        out.push(best)
        taken[best] = 1
    return out

# --------- Core accuracies ---------
fn accuracy_from_labels(preds: List[Int], targets: List[Int]) -> Float64:
    var n = len(targets)
    if n == 0: return 0.0
    var ok = 0
    for i in range(n):
        if preds[i] == targets[i]: ok += 1
    return Float64(ok) / Float64(n)

fn accuracy_from_scores(scores: List[List[Float64]], targets: List[Int]) -> Float64:
    var N = len(scores)
    if N == 0: return 0.0
    var preds = List[Int]()
    for n in range(N):
        preds.push(argmax1d_f(scores[n]))
    return accuracy_from_labels(preds, targets)

fn topk_accuracy(scores: List[List[Float64]], targets: List[Int], k: Int) -> Float64:
    var N = len(scores)
    if N == 0: return 0.0
    var ok = 0
    for n in range(N):
        var idxs = topk_indices_row(scores[n], k)
        var t = targets[n]
        var hit = False
        for j in range(len(idxs)):
            if idxs[j] == t: 
                hit = True
                break
        if hit: ok += 1
    return Float64(ok) / Float64(N)

# --------- Confusion matrix & derived accuracies ---------
fn confusion_matrix(targets: List[Int], preds: List[Int], num_classes: Int = -1) -> List[List[Int]]:
    var N = len(targets)
    var C = num_classes
    if C <= 0:
        var m = 0
        for i in range(N):
            var a = targets[i]; var b = preds[i]
            if a > m: m = a
            if b > m: m = b
        C = m + 1
        if C <= 0: C = 1
    var cm = zeros2d_i(C, C)  # rows=true, cols=pred
    for i in range(N):
        var t = targets[i]; var p = preds[i]
        if t < 0 or p < 0 or t >= C or p >= C: continue
        cm[t][p] += 1
    return cm

fn _sum_row_i(x: List[List[Int]], r: Int) -> Int:
    var c = 0
    if len(x) == 0: return 0
    var m = len(x[0])
    for j in range(m): c += x[r][j]
    return c

fn class_accuracy_from_cm(cm: List[List[Int]]) -> List[Float64]:
    var C = len(cm)
    var acc = zeros1d_f(C)
    for c in range(C):
        var tp = cm[c][c]
        var row = _sum_row_i(cm, c)
        acc[c] = _safediv(Float64(tp), Float64(row), 0.0)
    return acc

fn balanced_accuracy(targets: List[Int], preds: List[Int], num_classes: Int = -1) -> Float64:
    var cm = confusion_matrix(targets, preds, num_classes)
    var ca = class_accuracy_from_cm(cm)
    var C = len(ca)
    if C == 0: return 0.0
    var s = 0.0
    for i in range(C): s += ca[i]
    return s / Float64(C)

# --------- Running meters ---------
struct AccuracyMeter:
    var correct: Int
    var total: Int
    var value: Float64

    fn __init__(out self):
        self.correct = 0
        self.total = 0
        self.value = 0.0

    fn reset(mut self):
        self.correct = 0
        self.total = 0
        self.value = 0.0

    fn update_from_counts(mut self, correct_add: Int, total_add: Int):
        self.correct += correct_add
        self.total += total_add
        if self.total > 0:
            self.value = Float64(self.correct) / Float64(self.total)

    fn update_from_batch(mut self, preds: List[Int], targets: List[Int]):
        var n = len(targets)
        var ok = 0
        for i in range(n):
            if preds[i] == targets[i]: ok += 1
        self.update_from_counts(ok, n)

struct WindowedAccuracy:
    var window: Int
    var buf: List[Int]  # 0/1 outcomes
    var ptr: Int
    var count: Int      # number of filled slots (<= window)
    var sum: Int        # sum(buf)
    var value: Float64

    fn __init__(out self, window: Int):
        if window < 1: window = 1
        self.window = window
        self.buf = zeros1d_i(window)
        self.ptr = 0
        self.count = 0
        self.sum = 0
        self.value = 0.0

    fn reset(mut self):
        self.ptr = 0
        self.count = 0
        self.sum = 0
        for i in range(self.window): self.buf[i] = 0
        self.value = 0.0

    fn _push(mut self, ok: Int):
        var old = 0
        if self.count < self.window:
            self.buf[self.ptr] = ok
            self.sum += ok
            self.ptr = (self.ptr + 1) % self.window
            self.count += 1
        else:
            old = self.buf[self.ptr]
            self.buf[self.ptr] = ok
            self.sum = self.sum - old + ok
            self.ptr = (self.ptr + 1) % self.window
        if self.count > 0:
            self.value = Float64(self.sum) / Float64(self.count)

    fn update_from_batch(mut self, preds: List[Int], targets: List[Int]):
        var n = len(targets)
        for i in range(n):
            var ok = (1 if preds[i] == targets[i] else 0)
            self._push(ok)

# --------- Multilabel accuracies (thresholded) ---------
# targets: 0/1 labels, outputs: probabilities/logits -> thresholded
fn multilabel_micro_accuracy(outputs: List[List[Float64]], targets: List[List[Int]], threshold: Float64 = 0.5) -> Float64:
    var N = len(outputs)
    if N == 0: return 0.0
    var L = 0
    if len(outputs[0]) > 0: L = len(outputs[0])
    var correct = 0
    var total = 0
    for i in range(N):
        for j in range(L):
            var p = (1 if outputs[i][j] >= threshold else 0)
            if p == targets[i][j]: correct += 1
            total += 1
    if total == 0: return 0.0
    return Float64(correct) / Float64(total)

fn multilabel_subset_accuracy(outputs: List[List[Float64]], targets: List[List[Int]], threshold: Float64 = 0.5) -> Float64:
    var N = len(outputs)
    if N == 0: return 0.0
    var L = 0
    if len(outputs[0]) > 0: L = len(outputs[0])
    var ok = 0
    for i in range(N):
        var all_ok = True
        for j in range(L):
            var p = (1 if outputs[i][j] >= threshold else 0)
            if p != targets[i][j]: 
                all_ok = False
                break
        if all_ok: ok += 1
    return Float64(ok) / Float64(N)

# --------- Self-test ---------
fn _self_test() -> Bool:
    var ok = True

    # Core accuracy & top-k
    var targets = List[Int]([0,1,2,1,0,2])
    var scores = List[List[Float64]]()
    # correct classes have higher scores except one
    scores.push(List[Float64]([0.9,0.05,0.05]))
    scores.push(List[Float64]([0.2,0.6,0.2]))
    scores.push(List[Float64]([0.1,0.2,0.7]))
    scores.push(List[Float64]([0.3,0.4,0.3]))  # true=1
    scores.push(List[Float64]([0.8,0.1,0.1]))
    scores.push(List[Float64]([0.1,0.2,0.7]))
    var acc = accuracy_from_scores(scores, targets)
    ok = ok and (acc >= 0.83)  # 5/6 correct
    var top2 = topk_accuracy(scores, targets, 2)
    ok = ok and (top2 == 1.0)

    # Balanced accuracy (class recalls mean)
    var preds = List[Int]()
    for n in range(len(scores)): preds.push(argmax1d_f(scores[n]))
    var bal = balanced_accuracy(targets, preds, -1)
    ok = ok and (bal > 0.7)

    # AccuracyMeter
    var meter = AccuracyMeter()
    meter.update_from_batch(preds, targets)
    ok = ok and (meter.total == len(targets)) and (meter.value >= 0.83)
    meter.update_from_counts(0, 2)  # add two mistakes
    ok = ok and (meter.value < 1.0)

    # WindowedAccuracy
    var wmeter = WindowedAccuracy(4)
    # feed sequence of hits/misses: 1,1,0,1,0  -> last 4: 1,0,1,0 => 0.5
    var preds2 = List[Int]([0,1,0,1,0])
    var t = List[Int]([0,1,2,1,1])
    wmeter.update_from_batch(preds2, t)
    ok = ok and (wmeter.count == 4 or wmeter.count == 4 or wmeter.count == 5)  # after last push, window=4
    ok = ok and (wmeter.value <= 0.75 and wmeter.value >= 0.25)

    # Multilabel
    var outs = List[List[Float64]]()
    outs.push(List[Float64]([0.9, 0.2, 0.7]))
    outs.push(List[Float64]([0.1, 0.6, 0.4]))
    var yml = List[List[Int]]()
    yml.push(List[Int]([1,0,1]))
    yml.push(List[Int]([0,1,0]))
    var mic = multilabel_micro_accuracy(outs, yml, 0.5)
    var sub = multilabel_subset_accuracy(outs, yml, 0.5)
    ok = ok and (mic >= 0.83) and (sub >= 0.5)

    return ok

 